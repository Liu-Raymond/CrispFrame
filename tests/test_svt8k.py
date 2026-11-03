"""SVT-AV1 8K 级 preset 限制：阈值边界与自愈标志单测（纯逻辑，无 ffmpeg）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispframe.codec import SVT_8K_AREA, is_svt_8k_error, svtav1_fix_preset

fails = []


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)


# ---- svtav1_fix_preset 边界 ----
check("4K 3840x2160 不升档（8.3M）", svtav1_fix_preset(4, 3840, 2160) == (4, False))
check("5K 壁纸 5120x2880 不升档（14.7M，用户存量内容）",
      svtav1_fix_preset(4, 5120, 2880) == (4, False))
check("5888x2880 不升档（17.0M，实测 preset4 通过）",
      svtav1_fix_preset(4, 5888, 2880) == (4, False))
check("7680x4320 升档（33.2M，用户实际失败案例）",
      svtav1_fix_preset(4, 7680, 4320) == (5, True))
check("6000x4000 升档（24.0M，实测拒绝）",
      svtav1_fix_preset(4, 6000, 4000) == (5, True))
check("7680x2880 升档（22.1M，实测拒绝——宽高需同时大的反例不成立）",
      svtav1_fix_preset(4, 7680, 2880) == (5, True))
check("竖版 4320x7678 升档（面积对称）",
      svtav1_fix_preset(4, 4320, 7678) == (5, True))
check("7104x2880 升档（20.5M，实测 preset4 可用但保守上调，无害）",
      svtav1_fix_preset(4, 7104, 2880) == (5, True))
check("已是 preset5 不动", svtav1_fix_preset(5, 7680, 4320) == (5, False))
check("preset6 不动", svtav1_fix_preset(6, 7680, 4320) == (6, False))
check("面积恰在阈值（20M）升档", svtav1_fix_preset(4, 5000, 4000) == (5, True))
check("阈值下界附近不升档（19.9M）",
      svtav1_fix_preset(4, 5000, 3980)[1] is False)

# ---- is_svt_8k_error 标志 ----
svt_line = "Svt[error]: 8k+ resolution support is limited to M5 and faster presets."
ffmpeg_line = ("[libsvtav1 @ 000002c7343950c0] Error setting encoder parameters: "
               "bad parameter (0x80001005)")
user_log = ("直通编码失败:\n[libsvtav1 @ 000002c7343950c0] Error setting encoder "
            "parameters: bad parameter (0x80001005)\n[vost#0:0/libsvtav1 @ ...] "
            "Error while opening encoder - maybe incorrect parameters")
check("SVT 原生错误行匹配", is_svt_8k_error(svt_line))
check("ffmpeg 包装层 libsvtav1+bad parameter 匹配", is_svt_8k_error(ffmpeg_line))
check("用户日志全文匹配", is_svt_8k_error(user_log))
check("无视频轨类错误不误判", not is_svt_8k_error(
    "无法读取视频流：文件可能没有视频轨"))
check("其他编码错误不误判（x264 空间不足）", not is_svt_8k_error(
    "编码器中断:\nNo space left on device"))
check("泛化 bad parameter（非 libsvtav1）不误判", not is_svt_8k_error(
    "Error setting encoder parameters: bad parameter"))

print()
print("阈值面积:", SVT_8K_AREA)
print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
sys.exit(1 if fails else 0)
