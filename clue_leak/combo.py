"""线索子集消融(控制变量法)工具:非空子集枚举。

对 m 条线索枚举全部 2^m-1 个非空子集,按 (大小, 字典序) 稳定排序 →
输出顺序确定,结果可按子集缓存/续跑。
"""
from itertools import combinations


def nonempty_subsets(m: int) -> list:
    """[(0,), (1,), ..., (0,1), ..., (0,...,m-1)]:大小升序、同大小字典序。"""
    return [s for size in range(1, m + 1) for s in combinations(range(m), size)]
