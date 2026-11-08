"""CrispFrame 命令行入口（存档压缩工具）。

用法：
  python -m crispframe input.mkv                        # 存档：SVT-AV1 crf30/preset4 → MKV
  python -m crispframe input.mkv --crf 27 --container mp4
  python -m crispframe input.mkv --affinity p           # 只用大核（或 e/0,2,4-7）
  python -m crispframe 源.mp4 --verify-only 成品.mkv     # 只验证已有成品

分段存档：长片按 600s 切段逐段直编；Ctrl-C 或中途退出会保留已完成分片，
重新运行同一命令自动从断点续跑；--restart 清除分片从头再来。
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import affinity
from .ffutils import probe
from .codec import available_containers, CodecSettings
from .pipeline import PassthroughCanceled
from .archive_job import parts_dir, run_segmented
from .verify import human_size, verify_output


def archive_pix_fmt_probe(src: Path):
    from .codec import archive_pix_fmt
    from .verify import probe_stream_meta
    return archive_pix_fmt(probe_stream_meta(src)["pix_fmt"])


def _parse_affinity(spec: str):
    """解析 --affinity：返回逻辑处理器列表；"auto" 返回 None（不干预）；
    解析失败返回哨兵 "bad"。"""
    s = (spec or "auto").strip().lower()
    if not s or s == "auto":
        return None
    try:
        p_set, e_set = affinity.topology()
    except Exception as e:  # noqa: BLE001
        print(f"错误: 本机 CPU 拓扑读取失败（{e}），--affinity 不可用", file=sys.stderr)
        return "bad"
    all_cpus = sorted(set(p_set) | set(e_set))
    if s == "all":
        return all_cpus
    if s == "p":
        if not p_set:
            print("提示: 本机无大核（非混合架构），--affinity p 不生效", file=sys.stderr)
        return p_set
    if s == "e":
        if not e_set:
            print("提示: 本机无小核（非混合架构），--affinity e 不生效", file=sys.stderr)
        return e_set
    ids: list[int] = []
    try:
        for token in s.split(","):
            token = token.strip()
            if "-" in token:
                a, b = token.split("-")
                ids.extend(range(int(a), int(b) + 1))
            else:
                ids.append(int(token))
    except ValueError:
        print(f"错误: 无法解析 --affinity {spec!r}（支持 auto/all/p/e/"
              f"逗号编号如 0,2,4-7）", file=sys.stderr)
        return "bad"
    ids = sorted(set(ids))
    bad = [i for i in ids if i not in all_cpus]
    if bad or not ids:
        print(f"错误: --affinity {spec!r} 中的逻辑处理器不存在"
              f"（本机: {','.join(map(str, all_cpus))}）", file=sys.stderr)
        return "bad"
    return ids


def main(argv=None) -> int:
    # GBK 控制台下日志符号（▶/✓/⚠）会让 print 直接崩——降级为 ? 而非退出
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass
    _cont_list = "、".join(sorted(available_containers("svtav1")))
    ap = argparse.ArgumentParser(
        prog="crispframe",
        description="开源视频归档压缩工具：SVT-AV1 分段直编（全流保留、色彩/时基跟随源、"
                    "断点续跑），编码后自动三项验证（体积/元数据/全量解码查错）")
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
    ap.add_argument("--affinity", default="auto",
                    help="CPU 亲和性：auto（不干预，默认）/ all（全开）/ p（仅大核）/ "
                         "e（仅小核）/ 逗号分隔逻辑处理器编号（如 0,2,4-7）")
    ap.add_argument("--restart", action="store_true",
                    help="忽略/清除已有未完成分片，从头开始存档")
    ap.add_argument("--shutdown", action="store_true",
                    help="存档成功并验证通过后 60 秒自动关机（shutdown /a 可取消）")
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
    pix = archive_pix_fmt_probe(args.input)
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

    # ---- 相关性解析 ----
    affinity_ids = _parse_affinity(args.affinity)
    if affinity_ids == "bad":
        return 2

    verify_tag = "编码后自动验证" if not args.no_verify else "验证已跳过"
    print(f"存档: SVT-AV1 crf{cs.crf}/preset{cs.preset} → {cs.container}，"
          f"分段直编（断点可续跑/暂停），全流保留（多音轨/字幕/章节），"
          f"色彩/时基跟随源，{verify_tag}")
    if args.restart:
        d = parts_dir(args.input, cs)
        if d.exists():
            for f in d.iterdir():
                f.unlink(missing_ok=True)
            d.rmdir()
            print("已清除未完成分片，从头开始存档")
        else:
            print("没有未完成分片，直接开始")
    t0 = time.perf_counter()
    try:
        result = run_segmented(
            info, args.input, out_path, cs,
            cancel_check=None,
            pause_check=None,
            progress_cb=lambda d, t: print(
                f"\r编码进度 {d:.0f}/{t:.0f}s", end="", flush=True),
            log_cb=lambda m: print(m, flush=True),
            on_process=(lambda proc: affinity.apply(proc, affinity_ids))
            if affinity_ids else None)
    except KeyboardInterrupt:
        print("\n已暂停（分片保留）。继续存档：重新运行同一命令即可从断点续跑；"
              "放弃：加 --restart")
        return 130
    except PassthroughCanceled:
        print("\n已取消")
        return 130
    except Exception as e:
        print(f"\n[错误] 存档失败（分片与残件已删除，源片保留）: {str(e)[:500]}",
              file=sys.stderr)
        return 1
    finally:
        print()
    if result == "pause":
        print("已暂停（分片保留）。继续存档：重新运行同一命令即可从断点续跑")
        return 130
    if result != "ok":
        return 1
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
            shutil.rmtree(parts_dir(args.input, cs), ignore_errors=True)
            print("结论: ✓ 验证全部通过，可归档")
            if args.shutdown:
                print("60 秒后自动关机（取消：shutdown /a）")
                subprocess.run(["shutdown", "/s", "/t", "60"], capture_output=True)
            return 0
        # 未过验证 = 不可归档：删除成品与分片（分片重编也无法消除坏段）
        out_path.unlink(missing_ok=True)
        shutil.rmtree(parts_dir(args.input, cs), ignore_errors=True)
        print("\n验证未通过，产物已删除（源片保留），不要删除源片！")
        return 1
    shutil.rmtree(parts_dir(args.input, cs), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
