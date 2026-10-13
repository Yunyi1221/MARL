"""运筹学经典基线：base-stock（订到 S）和 (s, S) 策略。

这些策略各仓库独立决策、不考虑供应商产能，是 MARL 需要超过的参照线。
用法：python -m ormat.baselines
"""
import argparse
import math

import numpy as np

from ormat.envs import EnvConfig, ReplenishmentEnv


def poisson_quantile(mean, q):
    """Poisson(mean) 的 q 分位数（不依赖 scipy）。"""
    k, p, cdf = 0, math.exp(-mean), math.exp(-mean)
    while cdf < q:
        k += 1
        p *= mean / k
        cdf += p
    return k


class BaseStockPolicy:
    """库存位置低于 s 时订到 S；s = S 时就是 base-stock 策略。"""

    def __init__(self, S, s=None, max_order=12):
        self.S = np.asarray(S, dtype=np.float64)
        self.s = self.S if s is None else np.asarray(s, dtype=np.float64)
        self.max_order = max_order

    def act(self, env: ReplenishmentEnv):
        pos = env.inventory_position()
        order = np.where(pos < self.s, self.S - pos, 0.0)
        return np.clip(np.round(order), 0, self.max_order).astype(np.int64)


class PriorityBaseStockPolicy(BaseStockPolicy):
    """考虑产能的启发式：总订单超过产能时，按缺货惩罚从高到低依次满足，低优先级的仓库少订。"""

    def __init__(self, S, priority, capacity, s=None, max_order=12):
        super().__init__(S, s, max_order)
        self.order = np.argsort(-np.asarray(priority))
        self.capacity = capacity

    def act(self, env: ReplenishmentEnv):
        want = super().act(env)
        out = np.zeros_like(want)
        left = np.full(want.shape[0], self.capacity)
        for i in self.order:
            out[:, i] = np.minimum(want[:, i], left)
            left -= out[:, i]
        return out


def critical_ratio_levels(cfg: EnvConfig):
    """报童模型：S_i = 覆盖 (lead_time+1) 期需求的 b/(b+h) 分位数。"""
    periods = cfg.lead_time + 1
    return [poisson_quantile(mu * periods, b / (b + h))
            for mu, h, b in zip(cfg.demand_mean, cfg.holding_cost, cfg.stockout_cost)]


def evaluate_policy(policy, cfg: EnvConfig, n_episodes=200, seed=12345):
    env = ReplenishmentEnv(cfg, n_envs=n_episodes, seed=seed)
    env.reset()
    total_cost = np.zeros(n_episodes)
    done = False
    while not done:
        _, _, done, info = env.step(policy.act(env))
        total_cost += info["cost"]
    return total_cost.mean(), total_cost.std()


def grid_search(cfg: EnvConfig, n_episodes=200):
    """以临界比例水平为中心，在统一偏移 delta 和再订货间隔 gap 上做网格搜索。"""
    S_cr = np.asarray(critical_ratio_levels(cfg), dtype=np.float64)
    best = None
    for delta in range(-4, 5):
        S = np.maximum(S_cr + delta, 0)
        for gap in (0, 1, 2, 4):
            policy = BaseStockPolicy(S, S - gap, cfg.max_order)
            mean, std = evaluate_policy(policy, cfg, n_episodes, seed=999)  # 搜索用独立种子
            if best is None or mean < best[0]:
                best = (mean, std, delta, gap, S)
    return best


def best_baselines(cfg: EnvConfig, n_episodes=500):
    """返回 {名称: (平均成本, 标准差)}，三种基线都在评估种子上计算。"""
    capacity = ReplenishmentEnv(cfg).capacity
    S_cr = critical_ratio_levels(cfg)
    out = {"base-stock（临界比例）": evaluate_policy(
        BaseStockPolicy(S_cr, max_order=cfg.max_order), cfg, n_episodes)}
    _, _, delta, gap, S = grid_search(cfg)
    out[f"(s,S) 网格搜索 delta={delta} gap={gap}"] = evaluate_policy(
        BaseStockPolicy(S, S - gap, cfg.max_order), cfg, n_episodes)
    best = None
    for d in range(-4, 5):
        S_d = np.maximum(np.asarray(S_cr) + d, 0)
        pol = PriorityBaseStockPolicy(S_d, cfg.stockout_cost, capacity, max_order=cfg.max_order)
        m, _ = evaluate_policy(pol, cfg, 200, seed=999)
        if best is None or m < best[0]:
            best = (m, d, pol)
    out[f"按优先级配给的 base-stock delta={best[1]}"] = evaluate_policy(best[2], cfg, n_episodes)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    args = parser.parse_args()

    cfg = EnvConfig(capacity_ratio=args.capacity_ratio)
    print(f"供应商产能 C = {ReplenishmentEnv(cfg).capacity}，平均总需求 = {sum(cfg.demand_mean)}")
    for name, (mean, std) in best_baselines(cfg, args.episodes).items():
        print(f"{name}: 每回合成本 {mean:.1f} ± {std:.1f}")


if __name__ == "__main__":
    main()
