"""极致画质档：调用 vendored SeedVR2（字节开源生成式视频修复，Apache-2.0）。

速度 ~0.2 帧/s（4K），只用于重要素材；快速档（--model *）仍是默认。
通过子进程调用上游 CLI，与其 Apache-2.0 代码保持进程隔离，升级互不影响。
"""
import subprocess
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "seedvr2"
_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"

# 默认：3B fp8 单步蒸馏版（16GB 显存可跑；7B 更强但需 blockswap）
DEFAULT_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def run_seedvr2(input_path: Path, output_dir: Path, resolution: int = 2160,
                dit_model: str = DEFAULT_DIT, batch_size: int = 1,
                color_correction: str = "wavelet_adaptive",
                max_frames: int | None = None, extra: list[str] | None = None) -> Path:
    if not _VENDOR.exists():
        raise FileNotFoundError(
            f"SeedVR2 未部署: {_VENDOR}（git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler 到该路径）")
    cmd = [
        str(_PYTHON), str(_VENDOR / "inference_cli.py"), str(input_path),
        "--output", str(output_dir), "--output_format", "mp4",
        "--dit_model", dit_model, "--resolution", str(resolution),
        "--batch_size", str(batch_size), "--attention_mode", "sdpa",
        "--vae_encode_tiled", "--vae_decode_tiled",
        "--color_correction", color_correction,
    ]
    if max_frames:
        cmd += ["--load_cap", str(max_frames)]
    if extra:
        cmd += extra
    env_preset = {"HF_ENDPOINT": "https://hf-mirror.com"}  # 国内镜像加速模型下载
    import os
    env = {**os.environ, **env_preset}
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"SeedVR2 退出码 {proc.returncode}")
    outs = sorted(output_dir.glob("*.mp4"))
    if not outs:
        raise RuntimeError("SeedVR2 未产出输出文件")
    return outs[0]
