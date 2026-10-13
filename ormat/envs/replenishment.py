"""多仓库补货环境（向量化，纯 numpy）。

N 个仓库共享一个产能有限的供应商。每个时间步：
  1. 每个仓库选择订货量 a_i ∈ {0, 1, ..., max_order}；
  2. 总订单超过供应商产能 C 时按比例配给（向下取整）；
  3. 到货进入 lead_time 步的在途队列，队首到货入库；
  4. 需求 ~ Poisson(mu_i) 到达，库存满足不了的部分记为缺货（lost sales）；
  5. 团队奖励 = -(持有成本 + 缺货成本 + 订货成本) 之和，所有仓库共享。

仓库之间是异构的（需求均值、缺货惩罚不同），产能 C 比平均总需求略高但不宽裕，
所以“谁多订、谁少订”需要协调，这正是自回归决策（MAT）想解决的问题。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvConfig:
    n_agents: int = 4
    horizon: int = 50
    lead_time: int = 2
    max_order: int = 12                      # 动作空间大小 = max_order + 1
    demand_mean: tuple = (2.0, 4.0, 6.0, 8.0)
    holding_cost: tuple = (1.0, 1.0, 1.0, 1.0)
    stockout_cost: tuple = (2.0, 4.0, 8.0, 16.0)  # 异构优先级：缺货惩罚越大越“重要”
    order_cost: float = 0.0
    capacity_ratio: float = 1.05             # 供应商产能 = ratio × 平均总需求
    init_inventory_ratio: float = 1.0        # 初始库存 = ratio × mu_i
    reward_scale: float = 0.01

    def __post_init__(self):
        for name in ("demand_mean", "holding_cost", "stockout_cost"):
            v = getattr(self, name)
            if len(v) != self.n_agents:
                raise ValueError(f"{name} 的长度 {len(v)} 与 n_agents={self.n_agents} 不一致")


class ReplenishmentEnv:
    """批量运行 n_envs 个独立环境副本。所有数组的第 0 维是环境，第 1 维是仓库。"""

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: int = 0):
        self.cfg = cfg
        self.n_envs = n_envs
        self.n_agents = cfg.n_agents
        self.n_actions = cfg.max_order + 1
        self.rng = np.random.default_rng(seed)

        self.mu = np.asarray(cfg.demand_mean, dtype=np.float64)
        self.h = np.asarray(cfg.holding_cost, dtype=np.float64)
        self.b = np.asarray(cfg.stockout_cost, dtype=np.float64)
        self.capacity = int(round(cfg.capacity_ratio * self.mu.sum()))

        # obs = [库存, 在途(lead_time), 上期需求, 上期实际到手订单, mu, h, b, 产能, 剩余时间]
        self.obs_dim = 1 + cfg.lead_time + 2 + 3 + 2
        self._mu_scale = float(self.mu.max())

    # ------------------------------------------------------------------ #
    def reset(self):
        E, N, L = self.n_envs, self.n_agents, self.cfg.lead_time
        self.t = 0
        self.inventory = np.tile(np.round(self.cfg.init_inventory_ratio * self.mu), (E, 1))
        self.pipeline = np.tile(np.round(self.mu)[:, None], (E, 1, L)).astype(np.float64)
        self.last_demand = np.tile(self.mu, (E, 1))
        self.last_filled = np.tile(self.mu, (E, 1))
        return self._obs()

    def step(self, actions):
        """actions: (n_envs, n_agents) int。返回 obs, reward(n_envs,), done(bool), info。"""
        cfg = self.cfg
        a = np.clip(np.asarray(actions), 0, cfg.max_order).astype(np.float64)

        # 1) 供应商按比例配给
        total = a.sum(axis=1, keepdims=True)
        scale = np.where(total > self.capacity, self.capacity / np.maximum(total, 1.0), 1.0)
        filled = np.floor(a * scale)

        # 2) 在途队列前移，队首到货
        arrived = self.pipeline[:, :, 0].copy()
        self.pipeline[:, :, :-1] = self.pipeline[:, :, 1:]
        self.pipeline[:, :, -1] = filled
        self.inventory += arrived

        # 3) 需求与缺货
        demand = self.rng.poisson(self.mu, size=(self.n_envs, self.n_agents)).astype(np.float64)
        sold = np.minimum(self.inventory, demand)
        lost = demand - sold
        self.inventory -= sold

        holding = (self.h * self.inventory).sum(axis=1)
        stockout = (self.b * lost).sum(axis=1)
        ordering = cfg.order_cost * filled.sum(axis=1)
        cost = holding + stockout + ordering

        self.last_demand = demand
        self.last_filled = filled
        self.t += 1
        done = self.t >= cfg.horizon
        info = {
            "cost": cost,
            "holding": holding,
            "stockout": stockout,
            "lost_units": lost.sum(axis=1),
            "rationed": (total[:, 0] > self.capacity).astype(np.float64),
        }
        return self._obs(), -cost * cfg.reward_scale, done, info

    # ------------------------------------------------------------------ #
    def _obs(self):
        E, N = self.n_envs, self.n_agents
        s = self._mu_scale
        cols = [
            self.inventory[..., None] / s,
            self.pipeline / s,
            self.last_demand[..., None] / s,
            self.last_filled[..., None] / s,
            np.broadcast_to(self.mu / s, (E, N))[..., None],
            np.broadcast_to(self.h / self.h.max(), (E, N))[..., None],
            np.broadcast_to(self.b / self.b.max(), (E, N))[..., None],
            np.full((E, N, 1), self.capacity / (s * N)),
            np.full((E, N, 1), 1.0 - self.t / self.cfg.horizon),
        ]
        return np.concatenate(cols, axis=-1).astype(np.float32)

    def inventory_position(self):
        """库存 + 在途，用于基线策略。"""
        return self.inventory + self.pipeline.sum(axis=-1)
