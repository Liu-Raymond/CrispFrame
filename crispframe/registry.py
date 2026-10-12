"""模型菜单：内置"自动推荐"映射，专家模式可点名具体权重。"""
from dataclasses import dataclass
from pathlib import Path

from .layout import ROOT

PRETRAINED_DIR = ROOT / "models" / "pretrained"


@dataclass
class ModelSpec:
    name: str
    path: Path
    scale: int
    role: str          # sr | denoise | dedeblur
    content: str       # real | anime | general


# 默认菜单：名字 → models/pretrained 下的权重。缺文件时在启动时报错并提示下载。
BUILTIN: dict[str, ModelSpec] = {
    # 真人摄影向 4x（SPAN 架构，快且省显存）
    "nomos8k": ModelSpec("nomos8k", PRETRAINED_DIR / "4xNomos8k_span_otf_strong_no_update_params.pth", 4, "sr", "real"),
    # 2x 通用真人（SPAN，LDL 保细节训练）：1080p→4K 正确选择（4x 模型会多算 4 倍再缩小）
    "nomosuni": ModelSpec("nomosuni", PRETRAINED_DIR / "2xNomosUni_span.safetensors", 2, "sr", "real"),
    # Nomos8k 弱去噪版：细节保留更强
    "nomos8k-weak": ModelSpec("nomos8k-weak", PRETRAINED_DIR / "4xNomos8k_span_otf_weak_no_update_params.pth", 4, "sr", "real"),
    # 锐利风格 4x（ESRGAN 架构，社区最流行的"清晰"系）
    "ultrasharp": ModelSpec("ultrasharp", PRETRAINED_DIR / "4x-UltraSharp.pth", 4, "sr", "general"),
    # 新一代细节生成系 4x（轻量 Compact）
    "clearreality": ModelSpec("clearreality", PRETRAINED_DIR / "4x-ClearRealityV1.pth", 4, "sr", "real"),
    # 通用 4x（ESRGAN 架构）
    "x4plus": ModelSpec("x4plus", PRETRAINED_DIR / "RealESRGAN_x4plus.pth", 4, "sr", "general"),
    # 动画向 2x（Compact 架构，极快）
    "anime": ModelSpec("anime", PRETRAINED_DIR / "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth", 2, "sr", "anime"),
    # H.264 压缩伪影修复 1x（超分前置）
    "deh264": ModelSpec("deh264", PRETRAINED_DIR / "deH264_SuperUltraCompact.safetensors", 1, "dedeblur", "real"),
    # 去噪 1x（超分前置）
    "scunet": ModelSpec("scunet", PRETRAINED_DIR / "scunet_color_real_psnr.pth", 1, "denoise", "real"),
    # SCUNet GAN 变体（更锐）：A/B 目测与 PSNR 无可见差异且 4K 慢 2.5 倍，已从 UI 移除；
    # 权重可从 huggingface.co/krnl/ScuNET 取回放入 models/pretrained 即恢复可选
    "scunet-gan": ModelSpec("scunet-gan", PRETRAINED_DIR / "scunet_color_real_gan.pth", 1, "denoise", "real"),
    # NAFNet SIDD width32（官方 SIDD 降噪模型轻量版，width64 4K 软解不可用已弃）：
    # 权重镜像 huggingface.co/nyanko7/nafnet-models。spandrel 主库原生支持
    "nafnet-sidd": ModelSpec("nafnet-sidd", PRETRAINED_DIR / "NAFNet-SIDD-width32.pth", 1, "denoise", "real"),
}


def available() -> list[str]:
    return [n for n, s in BUILTIN.items() if s.path.exists()]


def resolve(name: str) -> ModelSpec:
    if name not in BUILTIN:
        raise KeyError(f"未知模型 {name}；可用: {sorted(BUILTIN)}")
    spec = BUILTIN[name]
    if not spec.path.exists():
        raise FileNotFoundError(f"模型文件缺失: {spec.path}")
    return spec
