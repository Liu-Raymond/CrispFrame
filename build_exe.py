"""Nuitka 打包：把 CrispFrame 编译为可分发的原生 CrispFrame.exe（含全部功能）。

用法：
  python build_exe.py             # dev 版（带控制台，供自动化 E2E 测试）
  python build_exe.py --release   # 发布版（无控制台，双击即 GUI）

产物：build/gui.dist/ —— 自包含目录（编译代码 + torch/TRT 运行时 + 模型权重 +
ffmpeg + config），直接 zip 分发。TRT 引擎缓存（models/engines）机器相关，不入包，
目标机首次运行各模型会现场构建引擎（首个任务偏慢属预期）。

需要：MSVC 2022 BuildTools（Nuitka 经 vswhere 自动定位）、~20GB 空闲磁盘。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "build" / "gui.dist"

# 懒加载依赖（torch.onnx 现场导出引擎时才 import）：venv 里有就显式捆绑。
# 注意：onnxruntime 仅开发期诊断用过、代码零引用，不入包（省 ~300MB）
_OPTIONAL_PACKAGES = ["onnx", "onnxconverter_common", "einops", "safetensors"]

# 随包分发的权重（与 registry/质量对比定版一致；缺失项由 registry 文件存在性门控自动隐藏）。
# 日后新增权重：放 models/pretrained 后在此追加一行即可。
_DISTRIBUTED_WEIGHTS = [
    "2xNomosUni_span.safetensors",
    "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
    "4xNomos8k_span_otf_weak_no_update_params.pth",
    "deH264_SuperUltraCompact.safetensors",
    "scunet_color_real_psnr.pth",
    "NAFNet-SIDD-width32.pth",
    "codeformer.pth",
    "yunet_2023mar.onnx",
    "rife4.26.pkl",
]


def _stage_ffmpeg() -> None:
    """把本机 winget 装的 Gyan.FFmpeg 的 ffmpeg/ffprobe 拷进阶段目录（GPLv3，README 需注明来源）。
    不走 Nuitka include-data-dir：.exe 被归类为可执行文件而非数据，构建后直接拷贝更可靠。"""
    from crispframe.ffutils import ffmpeg, ffprobe
    stage = ROOT / "build" / "ffmpeg_bundle"
    stage.mkdir(parents=True, exist_ok=True)
    for name, src in (("ffmpeg.exe", ffmpeg()), ("ffprobe.exe", ffprobe())):
        dst = stage / name
        if not dst.exists():
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="无控制台发布版（默认 dev 带控制台）")
    args = ap.parse_args()

    _stage_ffmpeg()

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--msvc=latest",
        "--output-dir=build",
        "--output-filename=CrispFrame.exe",
        "--company-name=CrispFrame",
        "--product-name=CrispFrame",
        "--file-version=1.0.0.0",
        "--product-version=1.0.0.0",
        "--file-description=CrispFrame AI 视频画质增强",
        "--windows-console-mode=" + ("disable" if args.release else "force"),
        # PySide6 插件：Qt 插件/资源入包必须显式启用（4.3 不自动启用）；
        # multimedia 是对比器/预览 QMediaPlayer 的媒体后端，"sensible" 集不含它
        "--enable-plugin=pyside6",
        "--include-qt-plugins=sensible,multimedia",
        # torch：本项目不用 Torch JIT，显式关闭（反膨胀插件要求二选一）
        "--module-parameter=torch-disable-jit=yes",
        # einops 的 tests 目录会被跟随进来，纯编译耗时无运行时价值
        "--nofollow-import-to=einops.tests",
        # 自家代码（含函数内懒导入的 face/interp/cli）
        "--include-package=crispframe",
        # 模型生态
        "--include-package=spandrel", "--include-package=spandrel_extra_arches",
        # GPU 推理栈（真实包目录名；DLL 走 include-package-data，防 ctypes 动态加载漏包）
        "--include-package=torch",
        "--include-package-data=torch",
        "--include-package=tensorrt", "--include-package=tensorrt_bindings",
        "--include-package=tensorrt_libs",
        "--include-package-data=tensorrt_libs",
        "--include-package=cv2", "--include-package-data=cv2",
        # 模型权重：按定版清单入包（engines 机器相关不入包；淘汰权重不分发）
        *([f"--include-data-file={ROOT / 'models' / 'pretrained' / w}=models/pretrained/{w}"
           for w in _DISTRIBUTED_WEIGHTS]),
    ]
    for pkg in _OPTIONAL_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            continue
        cmd += [f"--include-package={pkg}"]
    cmd.append("gui.py")

    print("[build] nuitka:", " ".join(cmd[2:]), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[build] nuitka 失败：{proc.returncode}", file=sys.stderr)
        return proc.returncode

    # ffmpeg 拷入 dist（.exe 不适合走 Nuitka 数据管线）
    (DIST / "ffmpeg").mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        shutil.copy2(ROOT / "build" / "ffmpeg_bundle" / name, DIST / "ffmpeg" / name)

    exe = DIST / "CrispFrame.exe"
    print(f"[build] 完成: {exe}" if exe.exists() else "[build] 未见产物 exe", flush=True)
    return 0 if exe.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
