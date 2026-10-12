# CrispFrame

**开源 AI 视频画质增强 —— 让每一帧都锐利。**

超分 · 插帧 · 降噪 · 压缩伪影修复 · 人脸增强，本地 GPU 推理，免费无订阅。
Windows + NVIDIA 显卡；GUI 任务流 + 与 GUI 同源的命令行。

![CrispFrame](assets/icon.png)

## 它解决什么

把"模型怎么选、参数怎么配"的知识固化成**零配置默认值**，同时保留专家模式：

- **速度**：TensorRT fp16 引擎（按模型×分辨率缓存，首次构建后秒级加载），帧的
  预处理/后处理全部留在 GPU 上完成
- **质量**：2x 超分到 4K 是速度正解（4x 模型多算 4 倍再缩小）；tone-match 修正
  模型固有偏色；bt709 显式色彩管理杜绝 ffmpeg 默认 bt601 偏色
- **体积**：默认 SVT-AV1 软编码（crf30/preset4，存档级画质），插帧/超分内容下
  体积约为 NVENC 硬编的 1/6

## 功能与模型

| 模块 | 默认模型 | 备注 |
|---|---|---|
| 超分（真人） | NomosUni 2x (SPAN) | 1080p→4K 正解；备选 Nomos8k-weak 4x |
| 超分（动漫） | AnimeJaNai HD v3 2x (Compact) | 极快 |
| 插帧 | RIFE 4.26（UHD 模式自动启用） | 2x 帧率 |
| 降噪 | SCUNet PSNR / NAFNet SIDD width32 | 0-1 强度混合（≤0.005 不加载模型） |
| 压缩伪影修复 | deH264 SuperUltraCompact | 超分前置 |
| 人脸增强 | YuNet 检测 + CodeFormer 修复 | 只处理人脸区域，跨帧 EMA 抑制抖动 |

模型推理通过 [spandrel](https://pypi.org/project/spandrel/) 加载，社区 SPAN/ESRGAN/Compact 架构通吃；
权重不随仓库分发，放入 `models/pretrained/` 即自动出现在菜单（缺文件自动隐藏）。

## 实测性能（RTX 5070 Ti 16GB，TensorRT 后端）

| 管线 | 速度 |
|---|---|
| 真人 720p→4K（NomosUni 2x） | ~41 帧/s |
| 动漫 1080p→4K（AnimeJaNai 2x） | ~33 帧/s |
| 4K 插帧（RIFE 4.26 UHD） | ~25 帧/s |
| **全链 1080p→4K（超分+锐化+SVT-AV1 编码）** | **11.75-16.5 帧/s** |

1 小时 1080p 素材全链处理约 2.5 小时，挂机一晚的量级。

## 快速开始

### 方式一：便携版（推荐，无需 Python）

从 [Releases](../../releases) 下载 `CrispFrame-x.x.x-win64.zip`，解压后双击 `CrispFrame.exe`。
要求见包内 `使用说明.md`（Windows 10/11 + NVIDIA 显卡 + 最新驱动）。

### 方式二：从源码运行

```bash
# 1) 环境：Python 3.13 + CUDA 12.8 版 PyTorch（Blackwell 必需）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2) ffmpeg（含 NVENC/SVT-AV1）
winget install --id Gyan.FFmpeg -e

# 3) 模型权重放入 models/pretrained/（见下方来源表）

# 4) 运行
python gui.py                       # GUI
python -m crispframe 输入.mp4 --tone-match --sharpen 0.6 --target-res 3840x2160
python -m crispframe 输入.mp4 --content anime --interp 2   # 动漫 + 2x 插帧
python -m crispframe --list-models  # 查看模型清单与全部参数
```

### 模型权重来源

| 文件 | 来源 |
|---|---|
| 2xNomosUni_span.safetensors / 4xNomos8k_span_otf_weak…pth / deH264_SuperUltraCompact.safetensors | [OpenModelDB](https://openmodeldb.info)（按名字搜索） |
| 2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth | [the-database/mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai) |
| scunet_color_real_psnr.pth | [cszn/SCUNet](https://github.com/cszn/SCUNet) |
| NAFNet-SIDD-width32.pth | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)（镜像：[nyanko7/nafnet-models](https://huggingface.co/nyanko7/nafnet-models)） |
| codeformer.pth + yunet_2023mar.onnx | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) / [opencv_zoo](https://github.com/opencv/opencv_zoo) |
| rife4.26.pkl | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) |

## 自行打包分发

```bash
# 1) 编译启动器（需 MSVC BuildTools）
cd launcher && build_launcher.bat

# 2) 组装便携包（python-embed + venv 库 + 权重 + ffmpeg → zip）
python make_dist.py
```

产物为自包含目录：`CrispFrame.exe`（C 启动器）+ `runtime\`（嵌入式 Python +
推理栈 + 代码）+ `models\` + `ffmpeg\`。另附 Nuitka 编译备选脚本 `build_exe.py`。

## 目录结构

```
crispframe/          # 管线核心：解码 → stage 链（SR/降噪/…）→ 插帧 → 编码
  sr.py              #   spandrel 加载 + TRT fp16 + 批处理
  interp.py _rife/   #   RIFE 4.26（TRT 默认关闭，见提交 fdd0b2a 的排查记录）
  trt.py             #   ONNX 导出 → TRT 引擎构建/缓存
  pipeline.py        #   ffmpeg rawvideo 管道 + EncodeConfig
  codec.py modules.py profiles.py   # 编码设置/模块链/配置文件（GUI 业务层）
  layout.py          # 包根定位（源码布局 = 便携包布局）
gui.py gui_preview.py compare_view.py   # GUI v2（任务流/预览/双路对比器）
launcher/            # C 启动器 + 图标
make_dist.py build_exe.py              # 便携包组装 / Nuitka 备选
```

## 已知限制

- 无 CPU 回退：没有 NVIDIA 显卡无法使用（启动时会提示）
- Windows 优先：macOS/Linux 未适配（管线大部分跨平台，硬解/编码参数按 Windows 调优）
- TRT 引擎缓存机器相关：换显卡后删除 `models/engines/` 会自动重建

## 许可

本项目代码以 [MIT](LICENSE) 发布。

分发自包含包时请注意第三方组件各自许可：ffmpeg（GPLv3）、NVIDIA CUDA/TensorRT
（NVIDIA EULA）、PySide6（LGPLv3）、RIFE（MIT）、模型权重遵循各自发布页许可。
