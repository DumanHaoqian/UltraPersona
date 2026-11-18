# UltraPersona v2（分支 `qwen_edit_2511`）

用 Qwen-Image-Edit-2511 把照片里的人（1 到 5 人）放进 PolyU 校园背景，再转成指定的画风。
目标机器是一张 RTX 5090（32GB 显存）。原始 BF16 权重放不下，所以只支持 FP8 和 4-bit 两种量化。

## 流程

1. **输入分析**（`analyze.py`）：检测人脸数量、每张脸的大小、清晰度和侧脸角度，判断取景是近景、半身还是全身。没有人脸或超过 5 人时直接拒绝；脸太小、太模糊或侧脸时给出提醒。
2. **拼提示词**（`prompts.py`）：背景段（`config.yaml` 里每个背景的场景描述和站位）+ 人数段（按检测到的人数和取景自动选择措辞）+ 风格段。
3. **生成**（`qwen_edit.py`）：图 1 是用户照片，图 2 是背景。支持两种方法：
   - 一步：一次编辑同时完成合成和风格化；
   - 两步：先合成写实照片，再对合成结果做风格化。
   默认用一步（`config.yaml` 的 `default_mode`）；“3D 动画”风格单独设为两步（风格里的 `mode`）。依据见 `experiments/ONE_VS_TWO.md`。
4. **人脸检查**（`face_metrics.py`）：用 InsightFace 的 ArcFace 把每张脸变成 512 维向量，计算输出和输入的余弦相似度。多人时用匈牙利算法一对一配对，报告人数是否一致、每个人的相似度、最差那个人的相似度和认错人的次数。不合格就换随机种子重试。
5. **背景锁定**（`control.py`，默认开）：先不加控制生成一次，得到人的位置；再生成一次，人以外的区域每一步都锁到背景（艺术风格锁到先单独风格化的背景）。锁定后人数不对就改用第一次的结果。
6. **人脸放大重绘**（默认只在写实照片风格）：分数低的脸裁出放大到 1024，涂掉旧脸后按原照片重画，贴回后更像本人才保留。
   身份梯度引导也在 `control.py` 里，默认关闭。三种方法的实验结论见 `experiments/CONTROL.md`。
7. **One Last Kiss** 风格不用模型（`styles_cv.py`，已改成矩阵运算）。

## 量化配置

| 配置 | 图像主体 | 文本编码器 | 说明 |
|---|---|---|---|
| `nf4` | 4-bit NF4（bitsandbytes） | 4-bit NF4 | 默认；两部分都常驻显存 |
| `fp8` | FP8（torchao，按行缩放） | 4-bit NF4 | 画质优先 |
| `fp8_te_offload` | FP8 | BF16，用完移回内存 | 文本编码器不量化 |

```bash
python -m persona2.quantize --preset nf4     # 量化一次，存到 config 里的 quantized_dir
python -m persona2.quantize --preset fp8
python -m experiments.bench_quant --preset nf4   # 显存峰值、速度、人脸相似度
```

## 网页

```bash
uvicorn persona2.server:app --host 0.0.0.0 --port 7860
# 只调页面、不加载模型：PERSONA_NO_MODEL=1 uvicorn persona2.server:app --port 7860
```

## 实验

```bash
python tools/fetch_commons_testcases.py testcases 15   # 下载可再使用许可的照片（来源记录在 testcases/sources.csv）
python tools/bucket_testcases.py testcases             # 按人数分到 testcases/people_1 ... people_5
python -m experiments.calibrate_faces                  # 用 LFW 标定相似度阈值
python -m experiments.one_vs_two --run v1              # 一步 vs 两步
python -m experiments.score_style --run v1             # 风格强度
python -m experiments.report --run v1
python -m experiments.control_ablation --run c1 --variants base "lock:r=0.5,t=styled,ri=0.2" "guide:g=0.04" "repair:g=0"
python -m experiments.score_style --dir experiments/results/control/c1
python -m experiments.control_report --run c1
```

## 环境

WSL2 Ubuntu，Python 3.11，PyTorch（CUDA 12.8 版），diffusers（GitHub 最新版），torchao，bitsandbytes，
insightface + onnxruntime（只用 CPU）。模型权重放在 `/mnt/d/hf_models`。

许可证：Qwen-Image-Edit-2511 是 Apache 2.0。InsightFace 自带的 buffalo_l 人脸模型只允许非商业研究使用。
