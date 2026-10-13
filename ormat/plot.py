"""汇总 results/ 下的训练曲线，画图并打印最终成本表。

用法：python -m ormat.plot --results ormat/results
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from ormat.baselines import best_baselines
from ormat.envs import EnvConfig


def load(results_dir):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*_s*", "progress.csv"))):
        mode = os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[0]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            runs[mode].append(rows)
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="ormat/results")
    p.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    args = p.parse_args()

    cfg = EnvConfig(capacity_ratio=args.capacity_ratio)
    baselines = best_baselines(cfg, 200)  # 与 train.py 的评估回合数一致

    runs = load(args.results)
    print(f"{'方法':<22}{'最终评估成本':>14}{'种子数':>8}   各仓库平均决策位次")
    for name, (mean, _) in baselines.items():
        print(f"{name:<22}{mean:>14.1f}{'-':>8}")
    for mode, rs in runs.items():
        final = [float(r[-1]["eval_cost"]) for r in rs]
        ranks = np.mean([[float(r[-1][f"rank_{i}"]) for i in range(cfg.n_agents)] for r in rs], axis=0)
        print(f"{'MAT-' + mode:<22}{np.mean(final):>9.1f} ± {np.std(final):<5.1f}{len(rs):>5}   "
              f"{np.round(ranks, 2).tolist()}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过画图")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, rs in runs.items():
        n = min(len(r) for r in rs)
        x = np.array([int(r["env_steps"]) for r in rs[0][:n]])
        y = np.array([[float(row["eval_cost"]) for row in r[:n]] for r in rs])
        m, s = y.mean(0), y.std(0)
        ax.plot(x, m, label=f"MAT-{mode}")
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    labels = ["base-stock (critical ratio)", "(s,S) grid search", "priority base-stock"]
    for (mean, _), label, ls in zip(baselines.values(), labels, ("--", ":", "-.")):
        ax.axhline(mean, ls=ls, c="gray", label=label)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval cost per episode (lower is better)")
    ax.set_ylim(top=2 * max(m for m, _ in baselines.values()))
    ax.set_title(f"capacity ratio = {cfg.capacity_ratio}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.results, "curves.png")
    fig.savefig(out, dpi=150)
    print(f"曲线已保存到 {out}")


if __name__ == "__main__":
    main()
