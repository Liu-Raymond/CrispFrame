"""CrispFrame 命令行入口。

用法示例：
  python -m crispframe input.mp4                     # 自动：4K 超分 + NVENC AV1
  python -m crispframe input.mp4 --content anime     # 动画模型
  python -m crispframe input.mp4 --interp 2          # 同时 2x 插帧
  python -m crispframe input.mp4 --model x4plus --crf 26
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

from . import registry
from .modules import StrengthBlend
from .ffutils import probe
from .gpu_post import GpuPost
from .sr import SuperRes
from .codec import (available_containers, VIDEO_CONTAINERS, CodecSettings,
                    build_encode_config_kwargs)
from .pipeline import EncodeConfig, PassthroughCanceled, run, run_passthrough
from .verify import human_size, probe_stream_meta, verify_output


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
            # 定版基线：真人 NomosUni 2x / 动漫 AnimeJaNai（与 GUI auto 一致）
            name = "anime" if args.content == "anime" else "nomosuni"
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
                                    else (args.model if args.model != "auto" else "nomosuni"))
            total_scale *= spec.scale
        out_h = info.height * total_scale
        scale = 0.5 if out_h >= 1400 else 1.0
    return RIFEInterpolator(device=device, scale=scale, use_trt=not args.no_trt)


def _smooth_encode(info, level: str = "medium") -> EncodeConfig:
    """平滑模式编码配置 = 验收定版配方，参数锁定：
    插帧产生的 2x 帧流在编码侧做 tmix 时域混合（等效快门的运动补偿，抹平
    AI 生成内容的时域毛刺），再由 fps 滤镜回采源帧率。
    编码固定 SVT-AV1 preset6/crf24/tune0（硬编对插帧内容体积浪费 ~6 倍）。
    三档滤镜与 GUI 平滑模块共用 modules._SMOOTH_MIX。"""
    from .modules import _SMOOTH_MIX
    return EncodeConfig(
        vf=f"{_SMOOTH_MIX[level]},fps={info.fps:.6f},scale=out_color_matrix=bt709:out_range=tv",
        args=["-c:v", "libsvtav1", "-preset", "6", "-crf", "24",
              "-svtav1-params", "tune=0", "-pix_fmt", "yuv420p"],
    )


def _apply_smooth_mode(args) -> None:
    """锁定为定版配方：只做 RIFE 2x 插帧 + 混合回采，画质增强全关。"""
    args.interp = 2
    args.model = "off"
    args.denoise = None
    args.deh264 = None
    args.face = False
    args.tone_match = False
    args.target_res = None
    args.sharpen = 0.0
    args.no_trt = True   # RIFE 的 TRT GridSample 撕裂 bug 未解，停用走 torch bf16


def out_name(src: Path, args) -> str:
    if args.smooth:
        return f"{src.stem}.cf.smooth-{args.smooth_level}.mp4"
    tags = []
    if args.deh264:
        tags.append("deh264")
    if args.denoise:
        tags.append("dn")
    if args.model != "off":
        m = args.model if args.model != "auto" else ("anime" if args.content == "anime" else "nomosuni")
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
    ap.add_argument("--smooth", action="store_true",
                    help="平滑模式：RIFE 2x 插帧后 tmix 运动补偿混合、回采源帧率"
                         "（等效快门平滑，抹平 AI 生成内容的时域顿挫）。"
                         "锁定配方编码，忽略超分/降噪/锐化/编码器等选项")
    ap.add_argument("--smooth-level", choices=["light", "medium", "heavy"],
                    default="medium",
                    help="平滑程度（配 --smooth）：light=当前帧主导 / "
                         "medium=180° 快门（默认，验收定版）/ heavy=最强平滑")
    ap.add_argument("--crf", type=int, default=None,
                    help="感知质量：普通/平滑模式默认 28（NVENC cq 尺度）；"
                         "--archive 用 SVT-AV1 crf 默认 30（25-27 稳/30 甜点/32 省）")
    ap.add_argument("--preset", default=None,
                    help="普通/平滑模式：NVENC 预设 p1-p7（默认 p5）；"
                         "--archive：SVT-AV1 速度档 1-13（默认 4，越小越慢越省）")
    ap.add_argument("--archive", action="store_true",
                    help="存档模式：直通转码（AI 增强参数不可用），SVT-AV1 恒定画质"
                         "（默认 crf30/preset4），全流保留（多音轨/字幕/章节），"
                         "编码后自动三项验证（体积/元数据/全量解码查错）")
    ap.add_argument("--force", action="store_true", help="输出已存在时覆盖（存档模式）")
    _cont_list = "、".join(sorted(available_containers("svtav1")))
    ap.add_argument("--container", default=None,
                    help=f"存档输出容器（默认 mkv）。本机可选：{_cont_list}；"
                         "非 MKV 容器的字幕/多音轨兼容性有限（见清单说明）")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过存档模式的编码后验证（不推荐，未验证前勿删源片）")
    ap.add_argument("--verify-only", metavar="成品.mkv",
                    help="跳过编码，只对已有成品跑三项验证；此时位置参数可选作对照源"
                         "（python -m crispframe 源.mp4 --verify-only 成品.mkv）")
    ap.add_argument("--encoder", default="av1_nvenc", choices=["av1_nvenc", "hevc_nvenc"])
    ap.add_argument("--fp32", action="store_true", help="禁用 bf16 半精度（质量最优先）")
    ap.add_argument("--batch", type=int, default=0,
                    help="批处理帧数；0 = 自动（按输出分辨率：4K 取 1-2，低分辨率取 4-8）")
    ap.add_argument("--face", action="store_true", help="超分后对人脸区域做 CodeFormer 修复")
    ap.add_argument("--face-w", type=float, default=0.8,
                    help="面部保真度 0-1：低=更'AI 修复'（细节多但可能改变五官内容），高=更保真（默认 0.8）")
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

    # ---- --verify-only：只对已有成品跑验证，不编码（位置参数可选作对照源） ----
    if args.verify_only:
        if args.archive or args.smooth:
            ap.error("--verify-only 只做验证，不能与 --archive/--smooth 同用")
        target = Path(args.verify_only)
        if not target.exists():
            print(f"文件不存在: {target}", file=sys.stderr)
            return 2
        src = None
        if args.input:
            if not args.input.exists():
                print(f"对照源不存在: {args.input}", file=sys.stderr)
                return 2
            src = args.input
        print(f"验证（体积/元数据/全量解码查错）: {target}")
        vr = verify_output(target, src, strict_metadata=src is not None)
        for line in vr.lines:
            print("  " + line)
        for f in vr.failures:
            print("  " + f)
        if vr.ok:
            print("结论: ✓ 验证全部通过，可归档" if src else "结论: ✓ 解码零错误")
            return 0
        print("\n验证未通过，不要删除源片！" if src else "\n验证未通过！")
        return 1

    if not args.input:
        ap.error("需要输入视频路径（或 --list-models / --verify-only 查看用法）")
    if not args.input.exists():
        print(f"输入不存在: {args.input}", file=sys.stderr)
        return 2
    if args.archive:
        conflicts = [n for v, n in ((args.smooth, "--smooth"), (args.interp, "--interp"),
                                    (args.denoise, "--denoise"), (args.deh264, "--deh264"),
                                    (args.face, "--face"),
                                    (args.model != "auto", "--model"),
                                    (args.content != "real", "--content"),
                                    (args.tone_match, "--tone-match"),
                                    (bool(args.target_res), "--target-res"),
                                    (args.sharpen > 0, "--sharpen"),
                                    (args.encoder != "av1_nvenc", "--encoder")) if v]
        if conflicts:
            ap.error(f"--archive 是直通转码，不能与 {'/'.join(conflicts)} 同用")
        if args.max_seconds is not None:
            ap.error("--archive 是终态归档，不支持 --max-seconds 截断（会产生假归档）")

    # ---- 输出路径与存档预检 ----
    cs_archive: CodecSettings | None = None
    if args.archive:
        cs_archive = CodecSettings()          # 定版存档基线：svtav1/crf30/preset4/mkv
        spec = cs_archive.spec()
        if args.crf is not None:
            cs_archive.crf = max(spec.crf_range[0], min(spec.crf_range[1], args.crf))
        if args.preset is not None:
            p = args.preset[1:] if args.preset[:1] in ("p", "P") else args.preset
            cs_archive.preset = max(spec.preset_range[0],
                                    min(spec.preset_range[1], int(p)))
        if args.container:
            if args.container not in available_containers("svtav1"):
                usable = "、".join(sorted(available_containers("svtav1")))
                ap.error(f"容器 {args.container} 与 SVT-AV1 不兼容或本机不可用；"
                         f"可选：{usable}")
            cs_archive.container = args.container
        # 像素格式映射（存档原则：输出元数据与源一致，验证严格比对 pix_fmt）；
        # 4:2:2/4:4:4 等格式在此预检拒绝，不空转编码
        from .codec import archive_pix_fmt
        pix = archive_pix_fmt(probe_stream_meta(args.input)["pix_fmt"])
        if pix is None:
            print("错误: 源像素格式超出严格一致支持范围（仅 yuv420p / yuv420p10le "
                  "可保持输出元数据一致）。请用 ffmpeg 直编或先把源转为 4:2:0。",
                  file=sys.stderr)
            return 2
        cs_archive.bit_depth, _ = pix
        out_path = args.output or args.input.with_name(
            f"{args.input.stem}.cf.archive-crf{cs_archive.crf}{cs_archive.ext()}")
        out_path.parent.mkdir(parents=True, exist_ok=True)   # 预检前先确保目录存在
        if out_path.exists() and not args.force:
            print(f"错误: 输出已存在: {out_path}（--force 覆盖）", file=sys.stderr)
            return 2
        free = shutil.disk_usage(out_path.parent).free
        src_bytes = args.input.stat().st_size
        if free < src_bytes / 4:
            print(f"警告: 输出所在盘剩余空间仅 {human_size(free)}，"
                  f"建议至少留源片体积 25% 的余量")
    else:
        if not args.smooth and args.smooth_level != "medium":
            print("提示: --smooth-level 需与 --smooth 同用，已忽略", file=sys.stderr)
        out_path = args.output or args.input.with_name(out_name(args.input, args))
        out_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe(args.input)

    print(f"输入: {info.width}x{info.height} @ {info.fps:.3f}fps, {info.frames or '?'} 帧, {info.duration:.1f}s")

    if args.archive:
        verify_tag = "编码后自动验证" if not args.no_verify else "验证已跳过"
        print(f"存档模式: 单进程直编 SVT-AV1 crf{cs_archive.crf}/preset{cs_archive.preset}"
              f"，全流保留（多音轨/字幕/章节），色彩/时基跟随源，{verify_tag}")
        encode = EncodeConfig(**build_encode_config_kwargs(
            cs_archive, info.fps, False, map_all=True))
        t0 = time.perf_counter()
        try:
            run_passthrough(info, out_path, encode=encode,
                            progress_callback=lambda d, t: print(
                                f"\r编码进度 {d:.0f}/{t:.0f}s", end="", flush=True),
                            cancel_check=None)
        except PassthroughCanceled:
            print("\n已取消")
            return 130
        except Exception as e:
            # 编码失败：残件顶着最终存档命名落盘有误归档风险（下次需 --force），删除之
            out_path.unlink(missing_ok=True)
            print(f"\n[错误] 编码失败（残缺产物已删除，源片保留）: {str(e)[:500]}",
                  file=sys.stderr)
            return 1
        finally:
            print()
        wall = time.perf_counter() - t0
        if out_path.exists():
            print(f"编码完成: {wall:.0f}s / {human_size(out_path.stat().st_size)}")
    else:
        if args.smooth:
            _apply_smooth_mode(args)
            print(f"平滑模式: RIFE 2x 插帧 → tmix 时域混合（{args.smooth_level}）→ 回采源帧率"
                  "（定版配方，画质/编码器选项已忽略）")
        stages = build_stages(args, info)
        interpolator = build_interpolator(args, info)
        out_fps = info.fps * (2 if interpolator else 1)
        chain = " → ".join(
            [type(s).__name__ + (f"({getattr(s, 'spec').name})" if hasattr(s, "spec") else "") for s in stages]
            + ([type(interpolator).__name__] if interpolator else [])
        ) or "直通"
        print(f"处理链: {chain}")
        if args.smooth:
            print(f"输出: {out_path}  ({info.fps:.3f}fps 回采, libsvtav1 crf24 preset6)")
        else:
            print(f"输出: {out_path}  ({out_fps:.3f}fps, {args.encoder} cq{args.crf or 28})")

        t0 = time.perf_counter()
        stats = run(
            info, out_path, stages=stages, interpolator=interpolator,
            encode=_smooth_encode(info, args.smooth_level) if args.smooth else
            EncodeConfig(encoder=args.encoder, crf=args.crf if args.crf is not None else 28,
                         preset=args.preset or "p5"),
            batch_size=args.batch, max_seconds=args.max_seconds,
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"\n完成: {stats['frames_out']} 帧输出 / {stats['wall_s']:.1f}s 总耗时, "
              f"处理 {stats['process_fps']:.2f} 帧/s, 输出 {size_mb:.1f} MB")

    if args.archive and not args.no_verify:
        print("\n编码后验证（体积 / 元数据 / 全量解码查错）...")
        vr = verify_output(out_path, args.input, strict_metadata=True)
        for line in vr.lines:
            print("  " + line)
        for f in vr.failures:
            print("  " + f)
        if vr.ok:
            print("结论: ✓ 验证全部通过，可归档")
            return 0
        # 未过验证 = 不可归档：产物以最终存档命名留在盘上有误归档风险，删除之
        out_path.unlink(missing_ok=True)
        print("\n验证未通过，产物已删除（源片保留），不要删除源片！")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
