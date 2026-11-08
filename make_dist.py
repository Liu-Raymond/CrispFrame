r"""组装便携分发包：三层布局，嵌入式 Python runtime。

产物布局（build/CrispFrame/）：
  CrispFrame.exe        薄启动器（launcher/launcher.c 编译产物）
  runtime\              python-embed + PySide6 + app\（代码）
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
VERSION = "2.0.0"

APP_FILES = ["gui.py", "archive_page.py"]

def _copy_site_packages() -> None:
    """存档工具只依赖 PySide6：白名单拷贝（旧全量拷贝约 8GB → 约 650MB）。"""
    src = ROOT / ".venv" / "Lib" / "site-packages"
    keep = {"PySide6", "shiboken6"}      # PySide6 的运行时依赖闭包就这两项

    def ignore(directory, names):
        base = Path(directory)
        if base != src:
            return set()
        return {n for n in names if n != "__pycache__" and n not in keep}

    # 整目录重建：旧 stage 的 site-packages 若是 AI 时代的全量拷贝（约 8.5GB），
    # 增量 copytree 会把 torch/tensorrt 整栈带进 v2.0.0 包
    if SITE.exists():
        shutil.rmtree(SITE)
    print("[dist] 复制 site-packages 白名单（PySide6，约 650MB）…", flush=True)
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
    stdlib_zip = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    pth.write_text(f"{stdlib_zip}\n.\nLib\\site-packages\napp\nimport site\n", encoding="ascii")


def _copy_app() -> None:
    app = RUNTIME / "app"
    # 整目录重建：防止复用旧 stage 时把 AI 时代的死文件（compare_view 等）带进包
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)
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
        # v1.x stage 的 AI 残留（权重/config）不属于 v2.0.0 包
        for stale in (STAGE / "models", STAGE / "config"):
            if stale.exists():
                shutil.rmtree(stale)
                print(f"[dist] 已清理旧 stage 残留 {stale.name}/", flush=True)
    STAGE.mkdir(parents=True, exist_ok=True)

    _ensure_python_embed()
    _copy_site_packages()
    _copy_app()

    # ffmpeg + 启动器 + 说明
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

    # 体积报告
    def du(p: Path) -> float:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return total / 1e9
    print(f"[dist] runtime = {du(RUNTIME):.2f} GB, 总计 = {du(STAGE):.2f} GB", flush=True)

    # zip（bsdtar，系统自带；Compress-Archive 对 >4GB 不可靠）
    print("[dist] 打 zip（存储模式，约 1GB）…", flush=True)
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
