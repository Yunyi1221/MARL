"""求解所有版本并和"各仓库独立套用 (3,40)"对比。用法：python -m ormat.hw3.run"""
import json
import os
import time

import numpy as np

from ormat.hw3.models import (Params, evaluate_pair, evaluate_serial, independent_sS_policy,
                              naive_serial_policy, solve_pair, solve_serial, solve_single)


STATES = [(0, 0), (3, 3), (3, 10), (3, 25), (10, 3), (5, 5), (10, 10), (20, 20)]


def describe_pair(Y):
    """几个代表状态下的最优订货量 [仓库1, 仓库2]。"""
    return {f"({a},{b})": [int(Y[a, b, 0] - a), int(Y[a, b, 1] - b)] for a, b in STATES}


def describe_serial(Y):
    """几个代表状态 (零售店, 中心仓) 下的最优动作 [调货量, 中心仓向外订货量]。"""
    out = {}
    for r, w in STATES:
        ship = int(Y[r, w, 0] - r)
        out[f"({r},{w})"] = [ship, int(Y[r, w, 1] - (w - ship))]
    return out


def best_naive_serial(prm):
    """为串联启发式网格搜索参数：先搜中心仓 (ws,wS)，再搜零售店 (s,S)，按 V(0,0) 选。"""
    best = None
    for ws in range(0, 41, 4):
        for wS in range(max(ws, 8), 41, 4):
            v = evaluate_serial(naive_serial_policy(3, 40, ws, wS), prm)[0, 0]
            if best is None or v < best[0]:
                best = (v, 3, 40, ws, wS)
    _, _, _, ws, wS = best
    for s in range(0, 9):
        for S in range(10, 41, 2):
            v = evaluate_serial(naive_serial_policy(s, S, ws, wS), prm)[0, 0]
            if v < best[0]:
                best = (v, s, S, ws, wS)
    return best[1:]


def main(out_dir=os.path.join(os.path.dirname(__file__), "results")):
    os.makedirs(out_dir, exist_ok=True)
    prm = Params()
    V1, pi1 = solve_single()
    print(f"单仓库原版：V(0) = {V1[0]:.2f}，策略 (s,S) = (3,40)，前 6 个状态的订货量 {pi1[:6].tolist()}")
    summary = {"single": {"V0": float(V1[0]), "policy": pi1.tolist()}}

    for version in ("independent", "joint_fixed", "supplier", "storage", "serial"):
        t0 = time.time()
        if version == "serial":
            V, Y = solve_serial(prm)
            s_r, S_r, ws, wS = best_naive_serial(prm)
            Vh = evaluate_serial(naive_serial_policy(s_r, S_r, ws, wS), prm)
            heuristic = f"零售店 ({s_r},{S_r}) + 中心仓 ({ws},{wS})（网格搜索最优）"
        else:
            V, Y = solve_pair(version, prm)
            Vh = evaluate_pair(version, independent_sS_policy(version, prm=prm), prm)
            heuristic = "各仓库独立 (3,40)"
        gap = (Vh - V) / V * 100
        if version == "storage":  # 只统计可达状态
            ii, jj = np.meshgrid(np.arange(41), np.arange(41), indexing="ij")
            gap = np.where(ii + jj <= prm.storage_cap, gap, np.nan)
        print(f"\n[{version}] 用时 {time.time() - t0:.1f}s")
        print(f"  最优 V(0,0) = {V[0, 0]:.2f}   {heuristic} = {Vh[0, 0]:.2f}   差距 {gap[0, 0]:.2f}%"
              f"   全状态平均差距 {np.nanmean(gap):.2f}%  最大 {np.nanmax(gap):.2f}%")
        acts = describe_serial(Y) if version == "serial" else describe_pair(Y)
        print("  代表状态下的最优订货量：", acts)
        np.savez_compressed(os.path.join(out_dir, f"{version}.npz"), V=V, Y=Y, V_heuristic=Vh)
        summary[version] = {"V00": float(V[0, 0]), "V00_heuristic": float(Vh[0, 0]), "heuristic": heuristic,
                            "gap00_pct": float(gap[0, 0]), "gap_mean_pct": float(np.nanmean(gap)),
                            "gap_max_pct": float(np.nanmax(gap)), "actions": acts}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
