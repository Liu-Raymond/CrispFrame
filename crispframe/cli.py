"""CrispFrame 命令行入口。

用法示例：
  python -m crispframe input.mp4                     # 自动：4K 超分 + NVENC AV1
  python -m crispframe input.mp4 --content anime     # 动画模型
  python -m crispframe input.mp4 --interp 2          # 同时 2x 插帧
  python -m crispframe input.mp4 --model x4plus --crf 26
"""
import argparse
import sys
import time
from pathlib import Path

from . import registry
from .modules import StrengthBlend
from .ffutils import probe
from .pipeline import EncodeConfig, run
from .gpu_post import GpuPost
from .sr import SuperRes


def _auto_batch(spec, width: int, height: int) -> int:
    """按输出分辨率与模型体量定批：4K 输出下大模型取 1（显存压力），小模型（Compact 系）可 4。"""
    out_px = width * height * spec.scale * spec.scale
    weight_mb = spec.path.stat().st_size / 1e6
    if out_px >= 7_500_000:      # ≈4K 输出
        return 4 if weight_mb <= 15 else 1
    if out_px >= 3_500_000:      # ≈1440p
        return 4 if weight_mb <= 15 else 2
    return 4


def build_stages(args, info) -> list:
    stages = []
    use_amp = not args.fp32

    def resolve_for(name):
        if name == "auto":
            name = "anime" if args.content == "anime" else "nomos8k"
        return registry.resolve(name)

    if args.deh264 and args.deh264 > 0.005:   # 与 GUI build 的 ≤0.005 短路语义一致
        spec = registry.resolve("deh264")
        batch = args.batch or _auto_batch(spec, info.width, info.height)
        sr = SuperRes(spec, use_amp=use_amp, batch_size=batch,
                      compile_model=args.compile, use_trt=not args.no_trt)
        stages.append(StrengthBlend(sr, args.deh264))
    if args.denoise and args.denoise > 0.005:  # 同上
        spec = registry.resolve("scunet")
        batch = args.batch or _auto_batch(spec, info.width, info.height)
        sr = SuperRes(spec, use_amp=use_amp, batch_size=batch,
                      compile_model=args.compile, use_trt=not args.no_trt)
        stages.append(StrengthBlend(sr, args.denoise))
    if args.model != "off":
        spec = resolve_for(args.model)
        batch = args.batch or _auto_batch(spec, info.width, info.height)
        stages.append(SuperRes(spec, use_amp=use_amp, batch_size=batch,
                               compile_model=args.compile, use_trt=not args.no_trt,
                               tone_match=args.tone_match))
    if args.face:
        from .face import FaceEnhance
        stages.append(FaceEnhance(fidelity=args.face_w))
    if args.target_res or args.sharpen > 0:
        from .gpu_post import GpuPost
        # 默认目标分辨率：SR 输出原分辨率（无 resize 时不做缩放）
        w, _, h = (args.target_res or "").partition("x")
        if not w:
            spec = resolve_for(args.model) if args.model != "off" else None
            scale = spec.scale if spec else 1
            w, h = str(info.width * scale), str(info.height * scale)
        stages.append(GpuPost(int(w), int(h), sharpen=args.sharpen))
    return stages


def build_interpolator(args, info, device: str = "cuda"):
    if not args.interp:
        return None
    from .interp import RIFEInterpolator  # 延迟导入，权重/代码缺失不影响纯超分
    if args.rife_scale > 0:
        scale = args.rife_scale
    else:
        # auto：按"超分后"（即插帧实际处理的）分辨率判断 UHD 模式
        total_scale = 1
        if args.model != "off":
            spec = registry.resolve("anime" if args.content == "anime" and args.model == "auto"
                                    else (args.model if args.model != "auto" else "nomos8k"))
            total_scale *= spec.scale
        out_h = info.height * total_scale
        scale = 0.5 if out_h >= 1400 else 1.0
    return RIFEInterpolator(device=device, scale=scale, use_trt=not args.no_trt)


def out_name(src: Path, args) -> str:
    tags = []
    if args.deh264:
        tags.append("deh264")
    if args.denoise:
        tags.append("dn")
    if args.model != "off":
        m = args.model if args.model != "auto" else ("anime" if args.content == "anime" else "nomos8k")
        tags.append(m)
    if args.interp:
        tags.append(f"x{args.interp}interp" if args.interp != 2 else "interp")
    return f"{src.stem}.cf.{'+'.join(tags) or 'pass'}.mp4"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispframe", description="CrispFrame 开源视频画质增强")
    ap.add_argument("input", type=Path, nargs="?", default=None)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--model", default="auto",
                    help="auto | nomos8k | x4plus | anime | off（默认 auto 按内容选）")
    ap.add_argument("--content", choices=["real", "anime"], default="real",
                    help="auto 模型选择时的内容类型")
    ap.add_argument("--deh264", nargs="?", const=1.0, default=None, type=float,
                    metavar="强度", help="超分前做 H.264 压缩伪影修复（可选强度 0-1，默认 1.0；<1 与原帧混合）")
    ap.add_argument("--denoise", nargs="?", const=1.0, default=None, type=float,
                    metavar="强度", help="超分前去噪（可选强度 0-1，默认 1.0；<1 与原帧混合）")
    ap.add_argument("--interp", type=int, default=0, choices=[0, 2],
                    help="插帧倍率（2 = 帧率翻倍）")
    ap.add_argument("--crf", type=int, default=28, help="感知质量（av1 默认 28）")
    ap.add_argument("--preset", default="p5", help="NVENC 预设 p1-p7")
    ap.add_argument("--encoder", default="av1_nvenc", choices=["av1_nvenc", "hevc_nvenc"])
    ap.add_argument("--fp32", action="store_true", help="禁用 bf16 半精度（质量最优先）")
    ap.add_argument("--batch", type=int, default=0,
                    help="批处理帧数；0 = 自动（按输出分辨率：4K 取 1-2，低分辨率取 4-8）")
    ap.add_argument("--face", action="store_true", help="超分后对人脸区域做 CodeFormer 修复")
    ap.add_argument("--face-w", type=float, default=0.8,
                    help="面部保真度 0-1：低=更'AI 修复'（细节多但可能改变五官内容），高=更保真（默认 0.8）")
    ap.add_argument("--extreme", action="store_true",
                    help="极致画质档：改用 SeedVR2 生成式修复（~0.2 帧/s，质量超快速档，长片慎用）")
    ap.add_argument("--extreme-res", type=int, default=2160,
                    help="极致档输出高度（默认 2160=4K）")
    ap.add_argument("--target-res", default=None,
                    help="输出分辨率 WxH（如 3840x2160）。SR 后先缩放到该分辨率再锐化——"
                         "超必要分辨率交给播放器缩放会整体发软，建议对齐显示设备原生分辨率")
    ap.add_argument("--sharpen", type=float, default=0,
                    help="unsharp 锐化幅度 0-2（SR 会放大源的运动模糊；0.6-1.0 可恢复商业软件式的'锐'观感）")
    ap.add_argument("--tone-match", action="store_true",
                    help="把输出的亮度/对比度统计逐帧对齐输入源（修正部分 SR 模型固有的偏暗/发灰）")
    ap.add_argument("--no-trt", action="store_true", help="禁用 TensorRT 后端（回退 PyTorch）")
    ap.add_argument("--rife-scale", type=float, default=0,
                    help="RIFE 光流计算尺度；0 = auto（≥1440p 用 0.5，官方 UHD 模式，约省 4 倍算力）")
    ap.add_argument("--compile", action="store_true",
                    help="启用 torch.compile（实测仅 Compact 类小模型有 12-29%% 收益；"
                         "SPAN 类架构会自动回退且拖慢后续批次，故默认关闭）")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="只处理前 N 秒（测试/预览用）")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args(argv)

    if args.list_models:
        print("可用模型：")
        for n in registry.available():
            spec = registry.BUILTIN[n]
            print(f"  {n:10s} {spec.scale}x  {spec.role:8s} {spec.content:7s} {spec.path.name}")
        return 0

    if not args.input:
        ap.error("需要输入视频路径（或 --list-models 查看模型）")
    if not args.input.exists():
        print(f"输入不存在: {args.input}", file=sys.stderr)
        return 2
    out_path = args.output or args.input.with_name(out_name(args.input, args))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe(args.input)

    if args.extreme:
        from .extreme import run_seedvr2
        print(f"极致档（SeedVR2 生成式）：输出高度 {args.extreme_res}px，速度 ~0.2 帧/s，请耐心等待")
        result = run_seedvr2(args.input, out_path.parent, resolution=args.extreme_res,
                             max_frames=int(args.max_seconds * info.fps) if args.max_seconds else None)
        if args.output and result.resolve() != out_path.resolve():
            result.replace(out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"\n完成: {out_path.name} {size_mb:.1f} MB")
        return 0

    print(f"输入: {info.width}x{info.height} @ {info.fps:.3f}fps, {info.frames or '?'} 帧, {info.duration:.1f}s")
    stages = build_stages(args, info)
    interpolator = build_interpolator(args, info)
    out_fps = info.fps * (2 if interpolator else 1)
    chain = " → ".join(
        [type(s).__name__ + (f"({getattr(s, 'spec').name})" if hasattr(s, "spec") else "") for s in stages]
        + ([type(interpolator).__name__] if interpolator else [])
    ) or "直通"
    print(f"处理链: {chain}")
    print(f"输出: {out_path}  ({out_fps:.3f}fps, {args.encoder} cq{args.crf})")

    t0 = time.perf_counter()
    stats = run(
        info, out_path, stages=stages, interpolator=interpolator,
        encode=EncodeConfig(encoder=args.encoder, crf=args.crf, preset=args.preset),
        batch_size=args.batch, max_seconds=args.max_seconds,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n完成: {stats['frames_out']} 帧输出 / {stats['wall_s']:.1f}s 总耗时, "
          f"处理 {stats['process_fps']:.2f} 帧/s, 输出 {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
