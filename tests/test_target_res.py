"""超分目标分辨率（x2/x4 直出与 WxH）的独立单测（纯 CPU，秒级）。

运行：.venv/Scripts/python.exe tests/test_target_res.py
覆盖：resolve_target 三形态与容错 / target_scale / effective_model 选型 /
ModuleInstance coerce 边界（GUI/profile 注入防护）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispframe.modules import (ModuleInstance, effective_model,
                                resolve_target, target_scale)

CASES: list[tuple[str, bool]] = []


def check(name: str, ok: bool):
    CASES.append((name, ok))


def main() -> int:
    # resolve_target
    check("x2 解析", resolve_target("x2", 2560, 1440) == (5120, 2880))
    check("x4 解析", resolve_target("x4", 1280, 720) == (5120, 2880))
    check("大写 X2 容错", resolve_target("X2", 1920, 1080) == (3840, 2160))
    check("WxH 解析", resolve_target("3840x2160", 2560, 1440) == (3840, 2160))
    check("空 = 与源相同", resolve_target("", 1920, 1080) is None)
    check("非法值 = None", resolve_target("abc", 1920, 1080) is None)

    # target_scale（诊断/展示用）
    check("x4 倍数", target_scale("x4", 720) == 4)
    check("WxH 倍数", target_scale("7680x4320", 1440) == 3)
    check("缩小档夹取", target_scale("1920x1080", 2160) == 1)
    check("空 = 1", target_scale("", 1080) == 1)

    # effective_model 选型（auto）
    em = lambda tr, content="real", h=1440: effective_model(  # noqa: E731
        {"model": "auto", "content": content, "target_res": tr}, h)
    check("x2 real → 2x 系", em("x2") == "nomosuni")
    check("x4 real → 4x 系", em("x4") == "nomos8k-weak")
    check("x4 anime → AnimeJaNai（2x 增强 + 上采）", em("x4", "anime") == "anime")
    check("x2 720p → 2x 系", em("x2", "real", 720) == "nomosuni")
    check("自定义 WxH 大比例 → 4x 系", em("7680x4320") == "nomos8k-weak")
    check("4K 档 1440 源 → 2x 系", em("3840x2160") == "nomosuni")
    check("1080p 档 2160 源（缩小）→ 2x 系", em("1920x1080", "real", 2160) == "nomosuni")
    check("显式模型优先", effective_model(
        {"model": "anime", "content": "real", "target_res": "x4"}, 1440) == "anime")
    check("无源高回落 2x 系", effective_model(
        {"model": "auto", "content": "real", "target_res": "x4"}, 0) == "nomosuni")

    # coerce 边界（GUI/profile 注入防护：choices 外值回落默认）
    check("x2 合法保留", ModuleInstance(
        "sr", {"target_res": "x2"}).params["target_res"] == "x2")
    check("x9 非法回落默认", ModuleInstance(
        "sr", {"target_res": "x9"}).params["target_res"] == "3840x2160")
    check("x99 非法回落默认", ModuleInstance(
        "sr", {"target_res": "x99"}).params["target_res"] == "3840x2160")
    check("大写 X2 回落默认", ModuleInstance(
        "sr", {"target_res": "X2"}).params["target_res"] == "3840x2160")

    failed = [n for n, ok in CASES if not ok]
    for n, ok in CASES:
        if not ok:
            print(f"  ✗ {n}")
    print(f"{len(CASES) - len(failed)}/{len(CASES)} 通过"
          + (f"；失败: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
