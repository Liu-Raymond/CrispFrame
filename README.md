# CrispFrame

开源 AI 视频画质增强套件 —— 让每一帧都锐利。

**定位**：功能广度对齐 UniFab，质量目标对齐 Topaz Video AI，模型全部采用开源阵容，免费、本地、可控。

- **平台**：Windows（开发基准机 RAYMOND-PC，RTX 5070 Ti 16GB，Blackwell / NVENC AV1+HEVC）
- **命名**：CrispFrame = "每一帧都锐利"；2026-10-05 选定，软件领域查重干净

## 为什么做它

商业软件的痛点：

- **UniFab**：1080p 6GB 超分 4K 变 20GB、插帧后 60GB（编码码率失控）；插帧 17 小时（实现低效）；模型黑盒（只有两档预设）
- **Video2X / REAL Video Enhancer**：功能少、设置复杂、无面部增强串联、模型库停在上一代 GAN 超分
- **Topaz Video AI**：质量天花板但 $299 且无试用

CrispFrame 的答案：把"模型怎么选、参数怎么配"的知识固化成**零配置默认值**，同时保留专家模式。

## 核心管线

```
解码(ffmpeg/PyAV) → 去压缩伪影(deH26x) → 去噪(NAFNet/SCUNet)
  → 超分(真人 RealESRGAN/4xNomos8k · 动画 AnimeJaNai)
  → 面部增强(CodeFormer + YuNet 检测，区域修复) → 插帧(RIFE 4.26 / GIMM-VFI) / 防抖(vidstab)
  → 编码(NVENC AV1 视觉无损 25-35 Mbps)
```

体积与速度承诺：同样 6GB 1080p 源，4K60 输出约 15-25GB（而非 60GB）；一部 2 小时电影全流程数小时（而非 17 小时）。

## 质量两档制

| 档位 | 超分引擎 | 适用 |
|---|---|---|
| 快速档（默认） | Real-ESRGAN / 4xNomos8k / AnimeJaNai | 日常全量处理，速度优先 |
| 极致档（可选） | SeedVR2（质量对标 Topaz Starlight）/ FlashVSR（速度对标） | 细节重建、老片修复，慢 |

真人/动画按内容自动分流。

## 模型菜单（Topaz 阵容 → 开源对标）

| Topaz | 功能 | CrispFrame 用 |
|---|---|---|
| Starlight | 生成式旗舰 | SeedVR2 / FlashVSR |
| Proteus | 通用可调超分 | Real-ESRGAN + 可调锐化 |
| Iris | 人脸/主体 | CodeFormer / GFPGAN |
| Artemis | 压缩片修复 | deH264/deH26x 模型 |
| Nyx | 去噪 | NAFNet / SCUNet |
| Apollo | 慢动作插帧 | GIMM-VFI（任意倍率） |
| Chronos | 快速插帧 | RIFE 4.26 |
| Themis | 防抖 | ffmpeg vidstab |
| Face Recovery | 人脸恢复 | CodeFormer + 时域平滑 |
| SDR to HDR | 动态范围 | **暂缓**（开源无成熟方案） |

v2 预留：人声分离（UVR5/Demucs，开源业界最强）、背景移除（Robust Video Matting）、音频上混（AudioSR + 传统上混，效果打折）。

## 界面哲学（抄 Topaz 的作业）

- 拖入视频 → 选档位 → 勾任务（超分/插帧/面部/去噪…自由组合）→ 开始
- **片段预览 + 对比视图**（截几秒试渲染，前后/分屏滑块对比）——最优先复刻的交互
- **任务队列/批处理**、断点续跑、监视文件夹
- 模型名/参数默认配好，专家选项折叠隐藏
- CLI 与 GUI 同源（管线核心与界面分离）

## 路线图

- **M1** 命令行管线跑通快速档，出对比样片验收质量与速度 ← 当前阶段
- **M2** 接入 SeedVR2 极致档，与快速档/UniFab A/B 对比
- **M3** GUI（PySide6 或本地 Web）：片段预览、队列、专家模式

## 目录结构（规划）

```
crispframe/       # Python 包：管线核心
  pipeline/       # 解码、滤镜链、编码
  models/         # 各模型推理封装
  cli.py          # 命令行入口
docs/             # 设计文档、基准测试
tests/
```

## 依赖与环境（M1 安装清单）

- ffmpeg（独立目录，含 NVENC）
- Python 3.12 + PyTorch CUDA 12.8+（RTX 5070 Ti sm_120 必需）
- 模型约 1GB：RealESRGAN x4 / 4xNomos8k / AnimeJaNai / RIFE 4.26 / CodeFormer / NAFNet
- M2 增量：SeedVR2-3B FP8 约 5-10GB（16GB 显存可跑，BlockSwap + VAE tiling）

## 实测性能（RTX 5070 Ti，TensorRT 后端，2026-10-06）

| 管线 | 速度 | 备注 |
|---|---|---|
| 真人 720p→4K（Nomos8k SPAN） | 41.4 帧/s | TRT fp16 |
| 动漫 1080p→4K（AnimeJaNai） | 33.4 帧/s | TRT fp16 |
| 4K 插帧（RIFE 4.26 UHD） | 25.2 帧/s | TRT + 官方 scale=0.5 |
| **全链 720p→4K60（超分+插帧）** | **11.95 帧/s** | 输出 4K60fps |

长视频预估（1080p24 源，超分 4K + 插帧 48fps 全链）：1 小时素材约 2.5 小时，2 小时约 5 小时——挂机一晚。

关键实现注记：
- TensorRT 11 无 BuilderFlag.FP16，fp16 由 onnxconverter-common 在模型侧转换；GridSample 的 grid 需手动插 Cast 保持与 input 同 dtype
- TRT 输入必须是标准 NCHW contiguous（channels_last 内存直接喂 data_ptr 会得到乱序输出——已踩坑并修复）
- RIFE UHD 模式（scale=0.5）要求输入边长被 128 整除（pad_div 随 scale 调整）
- 帧的 BGR→RGB/归一化/pad/量化/contiguous 全在 GPU 做，CPU 只做裸 24MB 拷贝——这一项给 4K 插帧带来 2.8 倍
- torch.compile（Windows 需 triton-windows）仅对 Compact 小模型有收益，SPAN 类自动回退，默认关闭
- 引擎按 (模型,shape,精度) 缓存于 models/engines/，首次构建 1-2 分钟，之后秒级加载

## 面部增强（M1.5，已实现）

- **YuNet**（OpenCV DNN，232KB ONNX）人脸检测：降采样自适应（检测图短边 ~900px）、框坐标量化 + 跨帧 EMA 平滑抑制抖动
- **CodeFormer**（spandrel-extra-arches 加载，bf16）：只对人脸区域修复（512x512，fidelity 可调 `--face-w`，默认 0.7）
- 羽化混合在 512 空间完成再一次性 resize 贴回（4K 大图 float 混合是耗时大户，已规避）
- 性能：720p→4K 超分+面部 6.7 帧/s；全链（+4K60 插帧）5.1 帧/s
