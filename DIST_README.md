# CrispFrame — AI 视频画质增强

超分 / 插帧 / 降噪 / 压缩伪影修复 / 人脸增强，一键批量处理。
本包为**免安装绿色版**：无需 Python，解压即用。

## 系统要求

- Windows 10 / 11（64 位）
- **NVIDIA 显卡（RTX 系列推荐）+ 最新显卡驱动**——AI 功能依赖 CUDA，
  无 N 卡启动时会弹窗提示（本包不含 CPU 推理回退）
- 磁盘空间：包体约 10 GB，建议另留 10 GB 以上给输出文件与引擎缓存
- 显存：4K 处理建议 12 GB+（1080p 素材 8 GB 可用）

## 快速开始

1. 解压到任意**本地磁盘**目录（不要放网络盘、压缩包内直接运行，也**不建议**
   放 `C:\Program Files`——引擎缓存需要写入包内 `models\engines\` 目录）
2. 双击 `CrispFrame.exe`
3. 首页拖入视频 → 添加模块（超分/插帧/降噪…）→ ⚙ 输出设置 → 开始处理

**首次运行注意**：每个 AI 模型第一次使用时会在本机构建 TensorRT 加速引擎
（约 1-3 分钟/模型，一次性），之后秒级加载。这是正常现象，不是卡死。

## 命令行（可选）

```bat
:: 快速档全功能（超分 + 2x 插帧，输出名自动带模块标签）
runtime\python.exe app\gui.py --cli 输入视频.mp4 --model nomosuni --interp 2

:: 查看模型清单与全部参数
runtime\python.exe app\gui.py --cli --list-models
runtime\python.exe app\gui.py --cli --help
```

## 常见问题

- **杀毒软件报毒/删除 exe**：编译型启动器偶发被误报，将 CrispFrame
  目录加入杀软白名单即可。
- **程序闪退**：看包根目录下的 `crash.log`，把内容连同显卡型号反馈给作者。
- **输出属性/缩略图问题**：Windows 自带 MKV 属性支持不完整，建议输出 MP4，
  或安装 Icaros 补全 MKV 属性显示。
- **速度慢**：确认用的是 NVIDIA 独显；4K 任务建议"输出设置"里用 SVT-AV1
  软编（画质/体积最优，编码本身较慢属正常）。

## 目录结构

```
CrispFrame.exe        主程序（双击启动 GUI）
runtime\              Python 运行时 + AI 推理栈 + 程序代码（app\）
models\pretrained\    模型权重（超分/降噪/去压缩/人脸/RIFE 插帧）
models\engines\       TensorRT 引擎缓存（首次运行各模型时自动生成，可整目录删除）
ffmpeg\               内置 ffmpeg / ffprobe（免安装）
config\               输出设置预设（运行时生成）
crash.log             崩溃日志（异常时生成）
```

## 开源与许可说明

本软件分发包包含以下第三方组件，各以对应上游许可证为准：

| 组件 | 用途 | 许可 / 来源 |
|---|---|---|
| ffmpeg / ffprobe | 解码/编码/封装 | **GPLv3**，Gyan.FFmpeg full build（gyan.dev/ffmpeg），源码可于上述地址获取 |
| CPython | 运行时 | Python Software Foundation License |
| PySide6 (Qt) | 图形界面 | LGPLv3 |
| PyTorch / CUDA | 深度学习运行时 | BSD-3（含 NVIDIA CUDA 专有运行时，遵循 NVIDIA EULA） |
| NVIDIA TensorRT | 推理加速 | NVIDIA SDK EULA |
| SVT-AV1（ffmpeg 内置） | AV1 编码 | BSD-2 + Alliance licenses |
| RIFE v4.26 | 插帧 | MIT（Practical-RIFE） |
| spandrel / spandrel-extra-arches | 社区模型加载 | Apache-2.0 / MIT |
| 社区模型权重 | NomosUni / AnimeJaNai / Nomos8k / SCUNet / NAFNet / CodeFormer 等 | 遵循各模型发布页许可，仅随包分发作个人使用 |
