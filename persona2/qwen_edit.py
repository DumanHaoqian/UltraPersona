"""Load Qwen-Image-Edit-2511 with FP8 or 4-bit weights and run edits on an RTX 5090 (32GB).

BF16 does not fit: the image transformer alone is ~41GB, the text encoder (Qwen2.5-VL-7B) ~16GB.

Each preset is (transformer mode, text-encoder mode):
  nf4   transformer 4-bit NF4 (bitsandbytes)      + text encoder 4-bit NF4
  fp8   transformer FP8 weights (torchao)          + text encoder 4-bit NF4
  nf4_te8  transformer 4-bit NF4                   + text encoder FP8 weights (torchao)
Old variants kept for the record (both broken, see experiments/QUANT_NOTES.md):
  nf4_all  NF4 on every layer (noisy images);  fp8_dq  FP8 weights *and* activations (NaN -> black images)

In the working presets the precision-sensitive layers stay BF16: the input projections (img_in, txt_in),
the timestep embedding, the final norm and output projection. FP8 is weight-only: Qwen-Image has very
large activations in some layers, and casting activations to FP8 overflows.

quantize.py saves quantized weights to <quantized_dir>/<preset>/; load_pipeline() uses them if present.
"""
from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager

import torch

# preset -> (transformer mode, text encoder mode)
PRESETS = {
    "nf4": ("nf4", "nf4"),
    "fp8": ("fp8", "nf4"),
    "nf4_te8": ("nf4", "fp8"),
    "nf4_mix": ("nf4_mix", "nf4"),
    "nf4_all": ("nf4_all", "nf4"),
    "fp8_dq": ("fp8_dq", "nf4"),
}
TR_SKIP = ["img_in", "txt_in", "txt_norm", "time_text_embed", "norm_out", "proj_out"]
TE_SKIP = ["lm_head", "visual.merger", "visual.patch_embed"]


def set_vram_cap(gb: float):
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))


def _bnb_nf4(lib: str, skip: list[str] | None):
    if lib == "diffusers":
        from diffusers import BitsAndBytesConfig
    else:
        from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
                              llm_int8_skip_modules=skip)


def _torchao_fp8(lib: str, skip: list[str] | None, dynamic_act: bool = False):
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, PerRow

    cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()) if dynamic_act else Float8WeightOnlyConfig()
    if lib == "diffusers":
        from diffusers import TorchAoConfig
    else:
        from transformers import TorchAoConfig
    return TorchAoConfig(cfg, modules_to_not_convert=skip)


def _tr_qconfig(mode: str):
    return {
        "nf4": lambda: _bnb_nf4("diffusers", TR_SKIP),
        "nf4_all": lambda: _bnb_nf4("diffusers", None),
        "nf4_mix": lambda: _bnb_nf4("diffusers", TR_SKIP + ["img_mod", "txt_mod"]),
        "fp8": lambda: _torchao_fp8("diffusers", TR_SKIP),
        "fp8_dq": lambda: _torchao_fp8("diffusers", None, dynamic_act=True),
    }[mode]()


def load_transformer(model_path: str, mode: str, saved_dir: str | None = None):
    from diffusers import QwenImageTransformer2DModel

    if saved_dir and os.path.isdir(os.path.join(saved_dir, "transformer")):
        return QwenImageTransformer2DModel.from_pretrained(
            os.path.join(saved_dir, "transformer"), torch_dtype=torch.bfloat16,
            use_safetensors=mode.startswith("nf4"), device_map="cuda")
    # device_map="cuda": each weight goes to the GPU and is quantized there, so the 41GB BF16
    # transformer never has to sit in system RAM (WSL has ~31GB).
    model = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", quantization_config=_tr_qconfig(mode), torch_dtype=torch.bfloat16,
        device_map="cuda")
    if mode == "nf4_mix":
        # The per-block modulation layers (img_mod / txt_mod, ~6.8B params) are kept out of NF4; store them
        # as FP8 weights instead (bf16 would cost ~13.6GB).
        from torchao.quantization import Float8WeightOnlyConfig, quantize_

        quantize_(model, Float8WeightOnlyConfig(),
                  filter_fn=lambda m, name: isinstance(m, torch.nn.Linear) and ("img_mod" in name or "txt_mod" in name))
        torch.cuda.empty_cache()
    return model


def load_text_encoder(model_path: str, mode: str, saved_dir: str | None = None):
    from transformers import Qwen2_5_VLForConditionalGeneration

    if saved_dir and os.path.isdir(os.path.join(saved_dir, "text_encoder")):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(saved_dir, "text_encoder"), torch_dtype=torch.bfloat16, device_map={"": 0})
    qc = _bnb_nf4("transformers", TE_SKIP) if mode == "nf4" else _torchao_fp8("transformers", TE_SKIP)
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map={"": 0})


def load_pipeline(model_path: str, preset: str = "nf4", quantized_dir: str | None = None,
                  lightning_lora: str | None = None, use_saved: bool = True):
    from diffusers import QwenImageEditPlusPipeline

    tr_mode, te_mode = PRESETS[preset]
    saved = os.path.join(quantized_dir, preset) if (quantized_dir and use_saved) else None
    t0 = time.time()
    transformer = load_transformer(model_path, tr_mode, saved)
    text_encoder = load_text_encoder(model_path, te_mode, saved)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
    if lightning_lora:
        pipe.load_lora_weights(lightning_lora)
    pipe.vae.to("cuda")  # transformer and text encoder were loaded straight onto the GPU
    pipe.set_progress_bar_config(disable=True)
    print(f"[qwen_edit] preset={preset} loaded in {time.time() - t0:.0f}s, "
          f"GPU allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    return pipe


@contextmanager
def peak_vram():
    """Yields a dict filled with peak allocated / reserved GB for the block."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = {}
    t0 = time.time()
    yield out
    torch.cuda.synchronize()
    out["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    out["seconds"] = time.time() - t0


def edit(pipe, images, prompt: str, negative_prompt: str = " ", seed: int = 0, steps: int = 40,
         true_cfg_scale: float = 4.0):
    """images: list of PIL images. The output takes the aspect ratio of the last image."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(image=images, prompt=prompt, negative_prompt=negative_prompt, true_cfg_scale=true_cfg_scale,
               guidance_scale=1.0, num_inference_steps=steps, generator=gen, num_images_per_prompt=1)
    return out.images[0]


def free(pipe=None):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
