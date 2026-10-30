r"""组装便携分发包：UniFab 式三层布局，嵌入式 Python runtime。

产物布局（build/CrispFrame/）：
  CrispFrame.exe        薄启动器（launcher/launcher.c 编译产物）
  runtime\              python-embed + venv 库（torch/TRT/PySide6…）+ app\（代码）
  models\pretrained\    定版权重（engines 目标机首次运行现场构建；SEEDVR2 极致档已移除）
  ffmpeg\               ffmpeg / ffprobe
  使用说明.md

用法：python make_dist.py   （前置：launcher/CrispFrame.exe 已编译，见 launcher/build_launcher.bat）
"""
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "build" / "CrispFrame"
RUNTIME = STAGE / "runtime"
SITE = RUNTIME / "Lib" / "site-packages"
VERSION = "1.2.0"

APP_FILES = ["gui.py", "gui_preview.py", "archive_page.py", "compare_view.py"]

# 随包分发的权重（与 registry/质量对比定版一致；缺失项由 registry 存在性门控自动隐藏）
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

# site-packages 排除：onnxruntime 零引用；pip 无用；__pycache__ 可再生；
# torch 的 include/头文件与 .lib 链接器产物、test 目录运行时不需要（省约 1GB）


def _copy_site_packages() -> None:
    src = ROOT / ".venv" / "Lib" / "site-packages"
    if (SITE / "torch").exists() and (SITE / "PySide6").exists():
        print("[dist] site-packages 已就绪，跳过", flush=True)
        return

    def ignore(directory, names):
        drop = {n for n in names if n == "__pycache__"}
        base = Path(directory)
        if base == src:
            drop |= {n for n in names if n.startswith("onnxruntime") or n == "pip"}
        if base.name == "torch":
            drop |= {"include", "test", "include_toolbox"} & set(names)
        if base.name == "lib" and base.parent.name == "torch":
            drop |= {n for n in names if n.endswith((".lib", ".a"))}
        return drop

    print("[dist] 复制 site-packages（约 8GB，需要几分钟）…", flush=True)
    shutil.copytree(src, SITE, ignore=ignore, dirs_exist_ok=True)


def _ensure_python_embed() -> None:
    ver = "{}.{}.{}".format(*sys.version_info[:3])
    marker = RUNTIME / "python.exe"
    if marker.exists():
        return
    zip_path = ROOT / "build" / f"python-{ver}-embed-amd64.zip"
    if not zip_path.exists():
        url = f"https://www.python.org/ftp/python/{ver}/python-{ver}-embed-amd64.zip"
        print(f"[dist] 下载 python-{ver} embed …", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    print("[dist] 解压嵌入式 Python → runtime/ …", flush=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RUNTIME)
    # ._pth 默认隔离 import site；改为常规布局（启用 site + app 目录入 path）
    pth = next(RUNTIME.glob("python*._pth"))
    pth.write_text("python313.zip\n.\nLib\\site-packages\napp\nimport site\n", encoding="ascii")


def _copy_app() -> None:
    app = RUNTIME / "app"
    app.mkdir(parents=True, exist_ok=True)
    for f in APP_FILES:
        shutil.copy2(ROOT / f, app / f)
    dst_pkg = app / "crispframe"
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(ROOT / "crispframe", dst_pkg,
                    ignore=lambda d, names: {n for n in names if n == "__pycache__"})


def main() -> int:
    launcher = ROOT / "launcher" / "CrispFrame.exe"
    if not launcher.exists():
        print("缺 launcher/CrispFrame.exe —— 先在 launcher/ 下跑 build_launcher.bat", file=sys.stderr)
        return 1

    if STAGE.exists():
        print(f"[dist] 复用现有 stage {STAGE}（各步骤幂等）", flush=True)
    STAGE.mkdir(parents=True, exist_ok=True)

    _ensure_python_embed()
    _copy_site_packages()
    _copy_app()

    # 权重（定版清单）+ ffmpeg + 启动器 + 说明
    (STAGE / "models" / "pretrained").mkdir(parents=True, exist_ok=True)
    for w in _DISTRIBUTED_WEIGHTS:
        shutil.copy2(ROOT / "models" / "pretrained" / w, STAGE / "models" / "pretrained" / w)
    ffmpeg_stage = ROOT / "build" / "ffmpeg_bundle"
    if not all((ffmpeg_stage / n).exists() for n in ("ffmpeg.exe", "ffprobe.exe")):
        # 大清理后重建：从本机 ffmpeg 实际位置（ffutils 同款定位链）组装 bundle
        from crispframe.ffutils import ffmpeg as _ff, ffprobe as _fp
        ffmpeg_stage.mkdir(parents=True, exist_ok=True)
        for src_exe, name in ((_ff(), "ffmpeg.exe"), (_fp(), "ffprobe.exe")):
            shutil.copy2(src_exe, ffmpeg_stage / name)
        print(f"[dist] ffmpeg_bundle 已从 {_ff()} 重建", flush=True)
    (STAGE / "ffmpeg").mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        shutil.copy2(ffmpeg_stage / name, STAGE / "ffmpeg" / name)
    shutil.copy2(launcher, STAGE / "CrispFrame.exe")
    shutil.copy2(ROOT / "DIST_README.md", STAGE / "使用说明.md")
    (STAGE / "config" / "profiles").mkdir(parents=True, exist_ok=True)

    # 体积报告
    def du(p: Path) -> float:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return total / 1e9
    print(f"[dist] runtime = {du(RUNTIME):.2f} GB, 总计 = {du(STAGE):.2f} GB", flush=True)

    # zip（bsdtar，系统自带；Compress-Archive 对 >4GB 不可靠）
    print("[dist] 打 zip（存储模式，约 10GB，需几分钟）…", flush=True)
    zip_path = ROOT / "build" / f"CrispFrame-{VERSION}-win64.zip"
    if zip_path.exists():
        zip_path.unlink()
    r = subprocess.run(["tar", "-a", "-cf", str(zip_path), "-C", str(STAGE.parent), STAGE.name])
    if r.returncode != 0:
        print(f"[dist] tar 打包失败 exit={r.returncode}", file=sys.stderr)
        return r.returncode
    print(f"[dist] 完成: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
