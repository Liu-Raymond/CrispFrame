"""CrispFrame 命令行入口（存档压缩工具）。

用法：
  python -m crispframe input.mkv                        # 存档：SVT-AV1 crf30/preset4 → MKV
  python -m crispframe input.mkv --crf 27 --container mp4
  python -m crispframe 源.mp4 --verify-only 成品.mkv     # 只验证已有成品
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

from .ffutils import probe
from .codec import (available_containers, CodecSettings,
                    build_encode_config_kwargs, is_svt_8k_error,
                    svtav1_fix_preset)
from .pipeline import (EncodeConfig, PassthroughCanceled, SilentTermination,
                       run_passthrough)
from .verify import human_size, probe_stream_meta, verify_output


def main(argv=None) -> int:
    _cont_list = "、".join(sorted(available_containers("svtav1")))
    ap = argparse.ArgumentParser(
        prog="crispframe",
        description="开源视频归档压缩工具：SVT-AV1 直编（全流保留、色彩/时基跟随源），"
                    "编码后自动三项验证（体积/元数据/全量解码查错）")
    ap.add_argument("input", type=Path, nargs="?", default=None,
                    help="输入视频；--verify-only 时可选作对照源")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="输出路径（默认与源同目录 *.cf.archive-crfN.mkv）")
    ap.add_argument("--crf", type=int, default=None,
                    help="SVT-AV1 恒定画质（默认 30；25-27 稳 / 30 甜点 / 32 省）")
    ap.add_argument("--preset", default=None,
                    help="SVT 速度档 1-13（默认 4，越小越慢越省；8K 级源自动 ≥5）")
    ap.add_argument("--container", default=None,
                    help=f"输出容器（默认 mkv）。本机可选：{_cont_list}；"
                         "非 MKV 容器的字幕/多音轨兼容性有限（见清单说明）")
    ap.add_argument("--force", action="store_true", help="输出已存在时覆盖")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过编码后验证（不推荐，未验证前勿删源片）")
    ap.add_argument("--verify-only", metavar="成品.mkv",
                    help="跳过编码，只对已有成品跑三项验证；位置参数可选作对照源"
                         "（python -m crispframe 源.mp4 --verify-only 成品.mkv）")
    args = ap.parse_args(argv)

    # ---- --verify-only：只对已有成品跑验证，不编码（位置参数可选作对照源） ----
    if args.verify_only:
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
        ap.error("需要输入视频路径（--verify-only 查看验证用法）")
    if not args.input.exists():
        print(f"输入不存在: {args.input}", file=sys.stderr)
        return 2

    # ---- 存档预检 ----
    cs = CodecSettings()          # 定版存档基线：svtav1/crf30/preset4/mkv
    spec = cs.spec()
    if args.crf is not None:
        cs.crf = max(spec.crf_range[0], min(spec.crf_range[1], args.crf))
    if args.preset is not None:
        p = args.preset[1:] if args.preset[:1] in ("p", "P") else args.preset
        cs.preset = max(spec.preset_range[0], min(spec.preset_range[1], int(p)))
    if args.container:
        if args.container not in available_containers("svtav1"):
            usable = "、".join(sorted(available_containers("svtav1")))
            ap.error(f"容器 {args.container} 与 SVT-AV1 不兼容或本机不可用；"
                     f"可选：{usable}")
        cs.container = args.container
    # 像素格式映射（存档原则：输出元数据与源一致，验证严格比对 pix_fmt）；
    # 4:2:2/4:4:4 等格式在此预检拒绝，不空转编码
    from .codec import archive_pix_fmt
    pix = archive_pix_fmt(probe_stream_meta(args.input)["pix_fmt"])
    if pix is None:
        print("错误: 源像素格式超出严格一致支持范围（仅 yuv420p / yuv420p10le "
              "可保持输出元数据一致）。请用 ffmpeg 直编或先把源转为 4:2:0。",
              file=sys.stderr)
        return 2
    cs.bit_depth, _ = pix
    out_path = args.output or args.input.with_name(
        f"{args.input.stem}.cf.archive-crf{cs.crf}{cs.ext()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)   # 预检前先确保目录存在
    if out_path.exists() and not args.force:
        print(f"错误: 输出已存在: {out_path}（--force 覆盖）", file=sys.stderr)
        return 2
    free = shutil.disk_usage(out_path.parent).free
    src_bytes = args.input.stat().st_size
    if free < src_bytes / 4:
        print(f"警告: 输出所在盘剩余空间仅 {human_size(free)}，"
              f"建议至少留源片体积 25% 的余量")

    info = probe(args.input)
    print(f"输入: {info.width}x{info.height} @ {info.fps:.3f}fps, "
          f"{info.frames or '?'} 帧, {info.duration:.1f}s")

    verify_tag = "编码后自动验证" if not args.no_verify else "验证已跳过"
    # 8K 级分辨率（面积阈值见 codec.SVT_8K_AREA）SVT 拒绝 preset<5，主动上调
    svt_fixed, svt_bumped = svtav1_fix_preset(cs.preset, info.width, info.height)
    if svt_bumped:
        cs.preset = svt_fixed
        print(f"提示: {info.width}x{info.height} 为 8K 级分辨率，"
              f"SVT-AV1 自动上调 preset{cs.preset}")
    print(f"存档: 单进程直编 SVT-AV1 crf{cs.crf}/preset{cs.preset} → "
          f"{cs.container}，全流保留（多音轨/字幕/章节），色彩/时基跟随源，{verify_tag}")
    t0 = time.perf_counter()
    try:
        for attempt in range(2):
            encode = EncodeConfig(**build_encode_config_kwargs(cs, map_all=True))
            try:
                run_passthrough(info, out_path, encode=encode,
                                progress_callback=lambda d, t: print(
                                    f"\r编码进度 {d:.0f}/{t:.0f}s", end="", flush=True),
                                cancel_check=None)
                break
            except PassthroughCanceled:
                raise
            except Exception as e:
                # 编码失败：残件顶着最终存档命名落盘有误归档风险（下次需 --force），删除之
                out_path.unlink(missing_ok=True)
                if attempt == 0 and cs.preset < 5 and is_svt_8k_error(str(e)):
                    print(f"\n提示: SVT 拒绝该分辨率的 preset{cs.preset}"
                          f"（8K 级限制），自动改用 preset5 重试")
                    cs.preset = 5
                    continue
                if attempt == 0 and isinstance(e, SilentTermination):
                    print("\n提示: 编码进程被外部终止（无错误输出），自动重试一次")
                    continue
                raise
    except PassthroughCanceled:
        print("\n已取消")
        return 130
    except Exception as e:
        print(f"\n[错误] 编码失败（残缺产物已删除，源片保留）: {str(e)[:500]}",
              file=sys.stderr)
        return 1
    finally:
        print()
    wall = time.perf_counter() - t0
    if out_path.exists():
        print(f"编码完成: {wall:.0f}s / {human_size(out_path.stat().st_size)}")

    if not args.no_verify:
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
