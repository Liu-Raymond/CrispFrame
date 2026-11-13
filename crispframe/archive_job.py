"""分段存档编排：切片 → 逐段直编（断点续跑/暂停）→ concat 合并 → 三项验证。

分片目录 = {源同目录}/{stem}.cf.parts-crf{N}-{container}/：
  part{N:04d}.{ext}  每段成品（crf30 段长约 600s，段长常量 SEG_SECS）
  manifest.json      任务指纹（源/设置），续跑时校验；段进度记录
合并 = concat 流拷贝 + 从源片回挂章节/全局元数据；验证只对合并成品做
（全量解码才是真保证）。任何失败/放弃 → 成品与分片目录一起删（沿用存档
失败生命周期）；暂停 → 保留分片，重开软件加入同一源片自动续跑。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .codec import CodecSettings, build_encode_config_kwargs, is_svt_8k_error, svtav1_fix_preset
from .ffutils import VideoInfo, ffmpeg, probe, SUBPROC_FLAGS
from .pipeline import (EncodeConfig, PassthroughCanceled, SilentTermination,
                       run_passthrough)

SEG_SECS = 600        # 段长定版（秒）：段边界重编码代价与续跑粒度的平衡
SEEK_BACK = 15.0      # 双 -ss 的关键帧回退窗口


def parts_dir(src: Path, cs: CodecSettings) -> Path:
    return src.with_name(f"{src.stem}.cf.parts-crf{cs.crf}-{cs.container}")


def _manifest_path(d: Path) -> Path:
    return d / "manifest.json"


def _write_manifest(d: Path, fingerprint: dict, total: int) -> None:
    _manifest_path(d).write_text(json.dumps(
        {**fingerprint, "total_segments": total}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def fingerprint(src: Path, cs: CodecSettings, info: VideoInfo) -> dict:
    """任务指纹：任何影响输出的设置或源变化都会使旧分片失效。"""
    return {"src": str(src), "src_size": src.stat().st_size,
            "src_mtime": int(src.stat().st_mtime),
            "duration": round(info.duration, 3),
            "crf": cs.crf, "container": cs.container,
            "bit_depth": cs.bit_depth, "preset": cs.preset}


def _segment_done(part: Path, expect: float) -> bool:
    """完成段判定：能 probe 且时长与期望差 <1s（防半截文件冒充）。"""
    if not part.exists() or part.stat().st_size == 0:
        return False
    try:
        from .verify import probe_stream_meta
        probe_stream_meta(part)
        out = subprocess.run(
            [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", str(part), "-f", "null", "-"],
            capture_output=True, **SUBPROC_FLAGS)
        if out.returncode != 0:
            return False
        import json
        from .ffutils import ffprobe
        r = subprocess.run(
            [ffprobe(), "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(part)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", **SUBPROC_FLAGS)
        dur = float(json.loads(r.stdout)["format"]["duration"])
        return abs(dur - expect) < 1.0
    except Exception:  # noqa: BLE001
        return False


def run_segmented(
    info: VideoInfo,
    src: Path,
    out_path: Path,
    cs: CodecSettings,
    *,
    seg_secs: int = SEG_SECS,   # 段长（秒）；测试用小段
    cancel_check=None,          # -> True：立即终止当前段并放弃（删分片）
    pause_check=None,           # -> True：当前段跑完后暂停（保留分片）
    progress_cb=None,           # progress_cb(done_s, total_s)（整片口径）
    fps_cb=None,                # fps_cb(realtime_fps)：整片口径实时帧率（2s 节流）
    log_cb=None,                # log_cb(str)
    on_process=None,            # on_process(proc)：每段 ffmpeg 启动后回调
) -> str:
    """返回 "ok" | "pause" | "fail"。失败时成品与分片目录均已删除。"""

    def log(msg: str):
        if log_cb:
            log_cb(msg)

    # 统一绝对路径：concat 列表条目相对列表文件解析，混用相对路径会双拼
    src = Path(src).resolve()
    out_path = Path(out_path).resolve()
    seg_t0_all = [time.perf_counter()]   # 整片编码墙钟（实时 fps 分母；续跑=本会话口径）

    # ---- 8K 级 preset 主动上调（与单文件路径同款） ----
    svt_fixed, svt_bumped = svtav1_fix_preset(cs.preset, info.width, info.height)
    if svt_bumped:
        cs.preset = svt_fixed
        log(f"⚠ {info.width}x{info.height} 为 8K 级分辨率，SVT-AV1 自动上调 preset{cs.preset}")

    total_dur = info.duration
    n_segments = max(1, -(-int(total_dur * 10) // int(seg_secs * 10)))  # 向上取整
    pd = parts_dir(src, cs)
    fp = fingerprint(src, cs, info)

    # ---- 续跑判定 ----
    done: set[int] = set()
    if pd.exists():
        try:
            m = json.loads(_manifest_path(pd).read_text(encoding="utf-8"))
            same = all(m.get(k) == v for k, v in fp.items())
        except Exception:  # noqa: BLE001
            same = False
        if same and m.get("total_segments") == n_segments:
            for i in range(n_segments):
                seg_start = min(i * seg_secs, total_dur)
                seg_end = min((i + 1) * seg_secs, total_dur)
                if _segment_done(pd / f"part{i:04d}{cs.ext()}", seg_end - seg_start):
                    done.add(i)
            log(f"发现未完成存档：{len(done)}/{n_segments} 段有效，"
                f"{'将继续' if len(done) < n_segments else '全部有效，直接合并'}")
        else:
            log("分片目录存在但设置/源片已变化，忽略旧分片重新开始")
            for f in pd.iterdir():
                f.unlink(missing_ok=True)

    pd.mkdir(parents=True, exist_ok=True)
    _write_manifest(pd, fp, n_segments)

    # ---- 逐段编码 ----
    for i in range(n_segments):
        if cancel_check and cancel_check():
            _cleanup(pd, out_path)
            raise PassthroughCanceled()
        if pause_check and pause_check():
            return "pause"
        if i in done:
            continue
        seg_start = min(i * seg_secs, total_dur)
        seg_end = min((i + 1) * seg_secs, total_dur)
        seg_len = seg_end - seg_start
        part = pd / f"part{i:04d}{cs.ext()}"
        log(f"▶ 段 {i + 1}/{n_segments}（{seg_start:.0f}-{seg_end:.0f}s）")
        t0 = time.perf_counter()
        seg_frames = int(info.fps * seg_len)     # 段帧数（实时 fps 口径）
        last_fps_emit = [0.0]                    # fps 提示节流（2s 一次）
        try:
            for attempt in range(2):
                encode = EncodeConfig(**build_encode_config_kwargs(cs, map_all=True))
                base = (i * seg_secs) if i else 0    # 进度口径：整片时间
                try:
                    def on_progress(dn, tt, b=base):
                        if progress_cb:
                            progress_cb(min(b + dn, total_dur), total_dur)
                        # 实时 fps：整片已编码帧数 / 整片编码墙钟，2s 节流
                        if fps_cb and seg_frames:
                            done_frames = int(info.fps * (b + dn))
                            el = time.perf_counter() - seg_t0_all[0]
                            now = time.perf_counter()
                            if now - last_fps_emit[0] >= 2.0:
                                last_fps_emit[0] = now
                                fps_cb(done_frames / el if el > 0 else 0.0)

                    run_passthrough(
                        info, part, encode=encode,
                        start=seg_start, duration=seg_len,
                        progress_callback=on_progress,
                        cancel_check=cancel_check,
                        process_hook=on_process)
                    break
                except PassthroughCanceled:
                    raise
                except Exception as e:
                    part.unlink(missing_ok=True)
                    if pause_check and pause_check():
                        # GUI 暂停 = 终止当前段（上层 proc.terminate → 静默死亡）；
                        # 残段丢弃、完成段保留，恢复时重编本段
                        return "pause"
                    if attempt == 0 and cs.preset < 5 and is_svt_8k_error(str(e)):
                        log(f"⚠ SVT 拒绝该分辨率的 preset{cs.preset}"
                            f"（8K 级限制），自动改用 preset5 重试本段")
                        cs.preset = 5
                        _write_manifest(pd, fp, n_segments)
                        continue
                    if attempt == 0 and isinstance(e, SilentTermination):
                        log("⚠ 编码进程被外部终止（无错误输出），自动重试本段")
                        continue
                    raise
        except KeyboardInterrupt:
            part.unlink(missing_ok=True)
            return "pause"
        except PassthroughCanceled:
            _cleanup(pd, out_path)
            raise
        except Exception as e:
            _cleanup(pd, out_path)
            log(f"[错误] {src.name}: {e}")
            return "fail"
        log(f"  段 {i + 1} 完成（{time.perf_counter() - t0:.0f}s）")

    # ---- 合并（流拷贝 + 章节/全局元数据从源回挂） ----
    log("合并分段（流拷贝，秒级）…")
    list_path = pd / "parts.txt"
    # concat 条目相对列表文件所在目录解析 → 裸文件名最稳（与 parts.txt 同目录）
    list_path.write_text(
        "\n".join(f"file 'part{i:04d}{cs.ext()}'" for i in range(n_segments))
        + "\n", encoding="utf-8")
    cmd = [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(list_path), "-i", str(src),
           "-map", "0", "-c", "copy", "-map_metadata", "1", "-map_chapters", "1",
           # 每段边界的音频包量化（copy 下 ~21-35ms/包）在 concat 处线性累积，
           # 20 段实测 +0.7s → 严格验证(0.5s)必失败。钳制到源时长：
           # 漂移全部收在尾段包对齐上，中间边界 ≤35ms 不可感
           "-t", f"{info.duration:.3f}"]
    if out_path.suffix.lower() in (".mp4", ".mov", ".m4v", ".f4v", ".3gp"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out_path))
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **SUBPROC_FLAGS)
    if r.returncode != 0:
        out_path.unlink(missing_ok=True)
        _cleanup(pd, out_path)
        log(f"[错误] 合并失败: {(r.stderr or '')[-400:]}")
        return "fail"
    # 合并成功即清分片（验证失败重并无意义——坏段重编才会变好，那时分片已无价值）
    shutil.rmtree(pd, ignore_errors=True)
    return "ok"


def _cleanup(pd: Path, out_path: Path) -> None:
    """失败/取消生命周期：成品与分片目录一起删（源片保留）。"""
    out_path.unlink(missing_ok=True)
    if pd.exists():
        for f in pd.iterdir():
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            pd.rmdir()
        except OSError:
            pass
