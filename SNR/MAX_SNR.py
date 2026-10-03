def calculate_marl_reward(P, H, n0, z, a, BW):
    n_b, n_u, n_k = a.shape
    signal_power_set = np.zeros((n_k, n_u))
    for b in range(n_b):
        for u in range(n_u):
            if z[b][u]:
                for k in range(n_k):
                    if a[b][u][k]:
                        signal_power_set[k][u] += P[b][u][k] * H[b][u][k]
    interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
    interference_m = np.tile(interference_sum, signal_power_set.shape[1]) - signal_power_set + n0
    unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
    total_rate = BW * np.sum(unscale_rate_m) / 1e6  # Mbps
    return total_rate
import numpy as np
import math
from collections import defaultdict
from module.utils import load_env, check_constraint
import torch
def heuristic_max_snr(B, U, K, P, H, n0, N_b, N_rb):
    """
    启发式MAX-SNR算法实现

    参数:
    B: 基站集合 [b1, b2, ...]
    U: 用户集合 [u1, u2, ...]
    K: 资源块(RB)集合 [k1, k2, ...]
    P: 发射功率, 3D数组 P[b][u][k]
    H: 信道增益, 3D数组 H[b][u][k]
    n0: 噪声功率
    N_b: 每个基站可服务的最大用户数
    N_rb: 每个用户在每个基站可分配的最大RB数

    返回:
    z: 用户-基站关联矩阵 (|B|x|U|)
    a: RB分配矩阵 (|B|x|U|x|K|)
    total_rate: 系统和速率
    """

    # 初始化变量
    n_b = len(B)
    n_u = len(U)
    n_k = len(K)

    # 创建索引映射
    b_idx = {b: i for i, b in enumerate(B)}
    u_idx = {u: i for i, u in enumerate(U)}
    k_idx = {k: i for i, k in enumerate(K)}

    # 初始化决策变量
    z = np.zeros((n_b, n_u), dtype=int)  # 用户-基站关联
    a = np.zeros((n_b, n_u, n_k), dtype=int)  # RB分配

    # 辅助数据结构
    user_associated_bs = {u: None for u in U}  # 用户关联的基站
    base_station_user_count = {b: 0 for b in B}  # 基站当前关联用户数
    user_rb_count = {b: {u: 0 for u in U} for b in B}  # 用户在基站分配的RB数

    # 每个RB上激活的用户-基站对 (用于计算干扰)
    active_rb_assignments = {k: [] for k in K}  # (b, u)对列表

    total_rate = 0.0  # 系统和速率

    # 步骤1: 预计算所有三元组的SNR并排序
    candidate_list=[]
    for b in B:
        for u in U:
            for k in K:
                # 计算无干扰SNR (忽略其他用户干扰)
                snr = (P[b][u][k] * H[b][u][k]) / n0
                candidate_list.append((b, u, k, snr))

    # 按SNR降序排序
    candidate_list.sort(key=lambda x: x[3], reverse=True)

    # 步骤2: 迭代激活三元组
    for b, u, k, snr in candidate_list:
        # 约束检查
        # 1. 用户已关联其他基站
        if user_associated_bs[u] is not None and user_associated_bs[u] != b:
            continue

        # 2. 用户尚未关联当前基站且当前基站用户容量已满
        if user_associated_bs[u] is not None and user_associated_bs[u] != b and base_station_user_count[b] >= N_b:
            continue

        # 3. 用户在基站的RB分配已达上限
        if user_rb_count[b].get(u, 0) >= N_rb:
            continue

        # 保存当前RB状态用于可能的回退
        rb_current_assignments = active_rb_assignments[k].copy()
        prev_total_rate = total_rate

        # 临时激活该三元组
        if user_associated_bs[u] is None:
            user_associated_bs[u] = b
            base_station_user_count[b] += 1
        user_rb_count[b][u] = user_rb_count[b].get(u, 0) + 1
        active_rb_assignments[k].append((b, u))

        # 计算激活后的RB速率
        new_rb_rate = calculate_rb_rate(k, active_rb_assignments[k], P, H, n0)

        # 计算速率增量
        old_rb_rate = calculate_rb_rate(k, rb_current_assignments, P, H, n0)
        rate_increment = new_rb_rate - old_rb_rate

        # 接受条件: 速率严格增加
        if rate_increment > 0:
            # 正式更新决策变量
            z[b_idx[b]][u_idx[u]] = 1
            a[b_idx[b]][u_idx[u]][k_idx[k]] = 1
            total_rate += rate_increment
        else:
            # 回退临时修改
            user_rb_count[b][u] = user_rb_count[b].get(u, 0) - 1
            if user_rb_count[b][u] == 0:
                if user_associated_bs[u] == b:
                    user_associated_bs[u] = None
                    base_station_user_count[b] -= 1
            active_rb_assignments[k] = rb_current_assignments

    return z, a, total_rate


def calculate_rb_rate(k, active_assignments, P, H, n0):
    """
    计算特定RB上的总速率

    参数:
    k: 资源块
    active_assignments: 该RB上激活的(b,u)对列表
    P: 发射功率
    H: 信道增益
    n0: 噪声功率

    返回:
    该RB上的总速率
    """
    if not active_assignments:
        return 0.0

    rb_rate = 0.0
    for b, u in active_assignments:
        signal_power = P[b][u][k] * H[b][u][k]
        interference = 0.0

        # 计算来自同RB其他传输的干扰
        for b_prime, u_prime in active_assignments:
            if b_prime == b and u_prime == u:
                continue  # 自身信号
            interference += P[b_prime][u_prime][k] * H[b_prime][u_prime][k]

        # 计算SINR
        sinr = signal_power / (interference + n0)
        rb_rate += math.log2(1 + sinr)  # 使用log2获得以比特为单位的速率

    return rb_rate




# 主函数
def main():
    # 生成一个环境实例,并保存它
    import os
    from module.environment import Environment
    from module.utils import get_config
    import pickle

    os.chdir('..')
    sce = get_config('SNR/set.yaml')
    # "nUEs":
    # "nRBs":
    # "nMBS":
    env = Environment(sce)
    nue = f'{sce.nUEs}'
    nbs = f'{sce.nMBS}'
    nrb = f'{sce.nRBs}'

    filepath = f'envs/{nbs}BS{nue}UE{nrb}RB/'
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    # 保存环境示意图
    env.showmap(f'envs/{nbs}BS{nue}UE{nrb}RB/{nbs}BS{nue}UE{nrb}RB.png')
    # 保存环境instance
    save_path = f'envs/{nbs}BS{nue}UE{nrb}RB/{nbs}BS{nue}UE{nrb}RB.zip'
    with open(save_path, 'wb') as f:
        pickle.dump(env, f)
    
    # 生成算法输入
    B = list(range(len(env.BSs)))
    U = list(range(len(env.UEs)))
    K = list(range(env.sce.nRBs))

    # 构建功率矩阵P
    P = np.ones((len(B), len(U), len(K)))

    # 构建信道矩阵H
    H = env.PH

    # 噪声功率
    n0 = env.get_n0()

    # 约束参数
    N_b = 12  # 默认每个基站最多服务5用户
    N_rb = 3  # 用户最多可分配全部RB


    print(f"基站: {B}")
    print(f"用户: {U}")
    print(f"资源块: {K}")
    print(f"每个基站最大用户数: {N_b}")
    print(f"每个用户在每个基站最大RB数: {N_rb}")

    # 运行算法
    z, a, total_rate = heuristic_max_snr(B, U, K, P, H, n0, N_b, N_rb)
    check_constraint(z,a,N_b,N_rb)
    # 打印结果
    print("\n优化结果:")
    bw = float(sce.BW)  # 带宽，单位 Hz
    mbps = total_rate * bw / 1e6
    marl_reward = calculate_marl_reward(P, H, n0, z, a, bw)
    print(f"系统和速率: {total_rate:.2f} bps/Hz, {mbps:.2f} Mbps")
    print(f"MAT同源奖励: {marl_reward:.2f} Mbps")

    print("\n用户-基站关联:")
    for b_idx, b in enumerate(B):
        for u_idx, u in enumerate(U):
            if z[b_idx][u_idx]:
                print(f"用户 {u} 关联基站 {b}")

    print("\n资源块分配:")
    for b_idx, b in enumerate(B):
        for u_idx, u in enumerate(U):
            if z[b_idx][u_idx]:
                allocated_rbs = [k for k_idx, k in enumerate(K) if a[b_idx][u_idx][k_idx]]
                if allocated_rbs:
                    print(f"基站 {b} 分配资源块 {allocated_rbs} 给用户 {u}")
    print(f"关联的用户数量: {np.sum(z > 0)} / {env.nUEs}")

    print()

if __name__ == "__main__":
    main()