"""affinity 模块测试：拓扑解析 + 进程亲和设置/回读（本机真实调用）。"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispframe import affinity

fails = []


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)


def main() -> int:
    if sys.platform != "win32":
        print("非 Windows 平台：仅验证 no-op 路径")
        check("拓扑为空", affinity.topology() == ([], []))
        return 0

    p_cores, e_cores = affinity.topology()
    all_cpus = sorted(set(p_cores) | set(e_cores))
    check(f"拓扑非空（P={len(p_cores)} E={len(e_cores)}）", p_cores or e_cores)
    check("P/E 不相交", not (set(p_cores) & set(e_cores)))
    check("P/E 并集=全部逻辑处理器且连续",
          all_cpus == list(range(len(all_cpus))))

    # 掩码边界
    try:
        affinity._mask_for([])
        check("空掩码被拒绝", False)
    except ValueError:
        check("空掩码被拒绝", True)
    try:
        affinity._mask_for([64])
        check("超界（64）被拒绝", False)
    except ValueError:
        check("超界（64）被拒绝", True)

    # 真实进程：起一个长睡 python，设亲和 → 回读
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.3)   # 等进程就绪
        ids = all_cpus[:max(1, len(all_cpus) // 2)]
        ok = affinity.apply(proc, ids)
        if ok:
            mask = affinity.current_mask(proc)
            want = sum(1 << i for i in ids)
            check(f"亲和设置+回读一致（选 {len(ids)} 核）", mask == want)
            # 再设成另一半（模拟运行中切换）
            ids2 = all_cpus[len(all_cpus) // 2:]
            affinity.apply(proc, ids2)
            check("运行中切换生效", affinity.current_mask(proc) ==
                  sum(1 << i for i in ids2))
        else:
            print("  （apply 返回 False——环境不支持，跳过回读断言）")
    finally:
        proc.kill()
        proc.wait()

    print()
    print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
