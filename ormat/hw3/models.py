"""ORIE 5570 HW3 库存问题的单仓库原版和多仓库扩展（两个仓库，可精确动态规划求解）。

原版（与作业一致）：
  每期先订货 a（立即到货），库存水平 y = s + a ≤ 40；需求 d ~ p_d（0..7）；
  成本 = 50·1[a>0] + 1·a + 0.1·(y-d)^+ + 10·(d-y)^+，下一期库存 (y-d)^+，折扣 0.98。

多仓库版本（两个相同的仓库，需求独立同分布）：
  independent  两个仓库完全独立，成本相加（应退化为两份 (3,40)）
  joint_fixed  联合订货：只要有仓库订货就付一次大固定成本 50，每个下单的仓库再付小固定成本 10
  supplier     共享供应商：两个仓库每期合计订货量 ≤ 24
  storage      共享仓储：两个仓库的库存水平合计 ≤ 40（同一仓库里的两种商品）
  serial       两级串联：零售店从中心仓调货（调货固定成本 10，立即到达，不超过中心仓库存），
               中心仓向外部供应商订货（固定成本 50 + 单价 1，下一期到货）；
               持有成本零售店 0.1、中心仓 0.05，只有零售店面对需求
"""
from dataclasses import dataclass

import numpy as np

CAP = 40
P_D = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
GAMMA = 0.98
FIXED, VAR, HOLD, PRICE = 50.0, 1.0, 0.1, 10.0

VERSIONS = ("independent", "joint_fixed", "supplier", "storage", "serial")


@dataclass
class Params:
    minor_fixed: float = 10.0       # joint_fixed：每个下单仓库的小固定成本
    supplier_cap: int = 24          # supplier：每期合计订货上限
    storage_cap: int = 40           # storage：合计库存水平上限
    ship_fixed: float = 10.0        # serial：零售店调货固定成本
    wh_hold: float = 0.05           # serial：中心仓持有成本


def level_tables(cap=CAP, p_d=P_D, hold=HOLD, price=PRICE):
    """按订货后库存水平 y 预先算好：期望持有+缺货成本 L[y]，以及转移矩阵 T[y, s']。"""
    L = np.zeros(cap + 1)
    T = np.zeros((cap + 1, cap + 1))
    for y in range(cap + 1):
        for d, p in enumerate(p_d):
            L[y] += p * (hold * max(y - d, 0) + price * max(d - y, 0))
            T[y, max(y - d, 0)] += p
    return L, T


# ---------------------------------------------------------------------- #
# 单仓库原版
# ---------------------------------------------------------------------- #
def solve_single(tol=1e-6):
    L, T = level_tables()
    s = np.arange(CAP + 1)[:, None]
    y = np.arange(CAP + 1)[None, :]
    a = y - s
    order = np.where(a > 0, FIXED + VAR * a, 0.0)
    feasible = a >= 0
    V = np.zeros(CAP + 1)
    while True:
        G = L + GAMMA * T @ V                       # G[y]
        Q = np.where(feasible, order + G[None, :], np.inf)
        V_new = Q.min(1)
        if np.abs(V_new - V).max() < tol:
            return V_new, Q.argmin(1) - np.arange(CAP + 1)
        V = V_new


# ---------------------------------------------------------------------- #
# 两仓库：状态 (s1, s2)，决策等价于选订货后水平 (y1, y2) ≥ (s1, s2)
# ---------------------------------------------------------------------- #
def _order_cost_2(version, prm: Params):
    """返回 (C, F)：C[s1,s2,y1,y2] 订货成本，F 可行性掩码。形状 (41,41,41,41)。"""
    n = CAP + 1
    s1 = np.arange(n)[:, None, None, None]
    s2 = np.arange(n)[None, :, None, None]
    y1 = np.arange(n)[None, None, :, None]
    y2 = np.arange(n)[None, None, None, :]
    a1, a2 = y1 - s1, y2 - s2
    F = (a1 >= 0) & (a2 >= 0)
    o1, o2 = a1 > 0, a2 > 0
    if version == "joint_fixed":
        C = np.where(o1 | o2, FIXED, 0.0) + prm.minor_fixed * (o1.astype(float) + o2) + VAR * (a1 + a2)
    else:
        C = FIXED * (o1.astype(float) + o2) + VAR * (a1 + a2)
    if version == "supplier":
        F = F & (a1 + a2 <= prm.supplier_cap)
    if version == "storage":
        # 合计超过上限的状态不可达（库存只会减少）；为了让值迭代有定义，允许它们“不订货”
        F = F & ((y1 + y2 <= prm.storage_cap) | ((a1 == 0) & (a2 == 0)))
    return np.broadcast_to(C, (n,) * 4), np.broadcast_to(F, (n,) * 4)


def solve_pair(version, prm=Params(), tol=1e-6, max_iter=5000):
    """值迭代求精确最优。返回 V[s1,s2] 和最优订货后水平 Y[s1,s2] = (y1, y2)。"""
    assert version in VERSIONS and version != "serial"
    n = CAP + 1
    L, T = level_tables()
    C, F = _order_cost_2(version, prm)
    Lsum = L[:, None] + L[None, :]
    V = np.zeros((n, n))
    for _ in range(max_iter):
        G = Lsum + GAMMA * T @ V @ T.T              # G[y1,y2] = 期望成本 + 折扣后的期望未来值
        Q = np.where(F, C + G[None, None], np.inf).reshape(n, n, n * n)
        V_new = Q.min(-1)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    idx = Q.argmin(-1)
    return V_new, np.stack(np.unravel_index(idx, (n, n)), -1)


def evaluate_pair(version, Y, prm=Params(), tol=1e-8):
    """精确评估一个给定策略 Y[s1,s2] -> (y1,y2) 的折扣成本（解线性方程）。"""
    n = CAP + 1
    L, T = level_tables()
    C, F = _order_cost_2(version, prm)
    s1, s2 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    y1, y2 = Y[..., 0], Y[..., 1]
    if not F[s1, s2, y1, y2].all():
        raise ValueError("策略包含不可行动作")
    r = C[s1, s2, y1, y2] + L[y1] + L[y2]
    # P[(s1,s2) -> (t1,t2)] = T[y1,t1] * T[y2,t2]
    P = np.einsum("ia,ib->iab", T[y1.ravel()], T[y2.ravel()]).reshape(n * n, n * n)
    V = np.linalg.solve(np.eye(n * n) - GAMMA * P, r.ravel())
    return V.reshape(n, n)


def independent_sS_policy(version, s=3, S=40, prm=Params()):
    """把单仓库最优 (s,S) 直接套到每个仓库；遇到耦合约束时按比例/优先截断到可行。"""
    n = CAP + 1
    Y = np.zeros((n, n, 2), dtype=int)
    for s1 in range(n):
        for s2 in range(n):
            y = [S if s1 <= s else s1, S if s2 <= s else s2]
            a = [y[0] - s1, y[1] - s2]
            if version == "supplier" and sum(a) > prm.supplier_cap:
                # 按需求量比例分配产能
                tot = sum(a)
                a = [a[0] * prm.supplier_cap // tot, a[1] * prm.supplier_cap // tot]
            if version == "storage" and s1 + a[0] + s2 + a[1] > prm.storage_cap:
                room = max(prm.storage_cap - s1 - s2, 0)
                tot = sum(a)
                a = [a[0] * room // tot, a[1] * room // tot] if tot else a
            Y[s1, s2] = [s1 + a[0], s2 + a[1]]
    return Y


# ---------------------------------------------------------------------- #
# 两级串联：状态 (r, w) = (零售店库存, 中心仓库存)
# ---------------------------------------------------------------------- #
def solve_serial(prm=Params(), tol=1e-6, max_iter=5000):
    """决策：零售店订货后水平 y_r（调货量 y_r - r ≤ w），中心仓下期库存 w' = w - 调货 + 新订货。"""
    n = CAP + 1
    L, T = level_tables()
    r = np.arange(n)[:, None, None, None]
    w = np.arange(n)[None, :, None, None]
    yr = np.arange(n)[None, None, :, None]
    wn = np.arange(n)[None, None, None, :]
    ship = yr - r
    left = w - ship                     # 调货后中心仓剩余
    buy = wn - left                     # 向外部订货量（下期到货）
    F = (ship >= 0) & (left >= 0) & (buy >= 0)
    C = (np.where(ship > 0, prm.ship_fixed, 0.0) + np.where(buy > 0, FIXED + VAR * buy, 0.0)
         + prm.wh_hold * np.maximum(left, 0))
    C, F = np.broadcast_to(C, (n,) * 4), np.broadcast_to(F, (n,) * 4)
    V = np.zeros((n, n))
    for _ in range(max_iter):
        G = L[:, None] + GAMMA * T @ V              # G[y_r, w']
        Q = np.where(F, C + G[None, None], np.inf).reshape(n, n, n * n)
        V_new = Q.min(-1)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    idx = Q.argmin(-1)
    return V_new, np.stack(np.unravel_index(idx, (n, n)), -1)


def evaluate_serial(Y, prm=Params()):
    n = CAP + 1
    L, T = level_tables()
    r, w = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    yr, wn = Y[..., 0], Y[..., 1]
    ship = yr - r
    left = w - ship
    buy = wn - left
    assert (ship >= 0).all() and (left >= 0).all() and (buy >= 0).all()
    cost = (np.where(ship > 0, prm.ship_fixed, 0.0) + np.where(buy > 0, FIXED + VAR * buy, 0.0)
            + prm.wh_hold * left + L[yr])
    P = np.zeros((n * n, n * n))
    rows = np.arange(n * n)
    for t in range(n):
        P[rows, t * n + wn.ravel()] += T[yr.ravel(), t]
    V = np.linalg.solve(np.eye(n * n) - GAMMA * P, cost.ravel())
    return V.reshape(n, n)


def naive_serial_policy(s=3, S=40, ws=10, wS=40):
    """零售店用 (3,40) 从中心仓调货（能调多少调多少），中心仓用自己的 (ws, wS) 补货。"""
    n = CAP + 1
    Y = np.zeros((n, n, 2), dtype=int)
    for r in range(n):
        for w in range(n):
            ship = min(S - r, w) if r <= s else 0
            left = w - ship
            wn = wS if left <= ws else left
            Y[r, w] = [r + ship, wn]
    return Y
