import torch, diffusers, torchao, bitsandbytes, transformers, insightface, inspect
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()[-3:])
print(diffusers.__version__, torchao.__version__, bitsandbytes.__version__, transformers.__version__, insightface.__version__)
from diffusers import QwenImageEditPlusPipeline
print(list(inspect.signature(QwenImageEditPlusPipeline.__call__).parameters)[:22])
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow
l = torch.nn.Linear(256, 256, bias=False).cuda().bfloat16()
quantize_(l, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
print("fp8 ok", l(x).shape)
import bitsandbytes as bnb
q = bnb.nn.Linear4bit(256, 256, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
print("nf4 ok", q(x).shape)
