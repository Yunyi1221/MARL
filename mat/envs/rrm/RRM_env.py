import numpy as np
import gym
# from RRM import Environment
import numpy as np
from marlcustomeEnv import MarlCustomEnv4
from types import SimpleNamespace
from gym.spaces import Box
from mat.utils.local_search import local_search_improve, local_search_stepwise

class RRMEnv:
    def __init__(self, args):
        self.args = args

        self.sce = self._create_scenario_from_args(args)

        self.env = MarlCustomEnv4(self.sce)
        self.num_agents = self.env.BS_num
        self.n_agents = self.num_agents  # ShareDummyVecEnv 需要

        # 每个 agent 的真实动作维度
        self._agent_act_dims = [space.shape[0] for space in self.env.action_spaces]
        # 统一到最大维度
        self._max_act_dim = max(self._agent_act_dims)
        padded_action_space = Box(low=0.0,
                                  high=1.0,
                                  shape=(self._max_act_dim,),
                                  dtype=np.float32)
        self.action_space = [padded_action_space] * self.num_agents
        # obs 取每个 BS 的 len(UE_set)*nRB，pad 到最大值
        # 1) 记录每个 agent 的真实 obs 维度，以及最大维度
        ue_counts = [len(bs.UE_set) for bs in self.env.BSs]
        self.local_obs_dims = [count * self.sce.nRBs for count in ue_counts]
        
        # 使用固定的最大观测空间维度，确保环境实例间的一致性
        # 理论上每个BS最多可以关联的UE数量基于覆盖范围，实际中可能超过配置的Nb
        # 为了保险起见，使用所有UE数量作为上限
        theoretical_max_ues_per_bs = self.sce.nUEs  # 最极端情况：所有UE都在一个BS范围内
        self.max_local_obs_dim = theoretical_max_ues_per_bs * self.sce.nRBs

        # 2) 用 max_local_obs_dim 构造统一的 observation_space (2D: max_users x nRBs)
        max_users = self.max_local_obs_dim // self.sce.nRBs
        padded_obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_users, self.sce.nRBs),
            dtype=np.float32
        )
        self.observation_space = [padded_obs_space] * self.num_agents

        # 3) 同理记录共享 obs 的维度 (3D: num_agents x nUEs x nRBs)
        self.share_obs_dim = self.num_agents * self.sce.nUEs * self.sce.nRBs
        full_share_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.sce.nUEs, self.sce.nRBs),
            dtype=np.float32
        )
        self.share_observation_space = [full_share_space] * self.num_agents
        print("max_local_obs_dim:", self.max_local_obs_dim)


        self.step_count = 0

        # Local Search settings (defaults are safe/off if attrs absent)
        self.use_local_search = bool(getattr(args, 'use_local_search', False))
        self.ls_budget = int(getattr(args, 'ls_budget', 0) or 0)
        self.ls_k = int(getattr(args, 'ls_k', 1) or 1)
        # Early stop is enabled by default; can be disabled via flag
        self.ls_early_stop = not bool(getattr(args, 'ls_disable_early_stop', False))
        # Stepwise search across UA/RB and reduced frequency
        self.ls_stepwise = bool(getattr(args, 'ls_stepwise', False))
        self.ls_include_rb = bool(getattr(args, 'ls_include_rb', True))
        # Apply LS every N steps (1 = every step)
        self.ls_interval = int(getattr(args, 'ls_interval', 1) or 1)

    def _create_scenario_from_args(self, args):
        sce = SimpleNamespace(
            nUEs = args.n_ues,
            nRBs = args.n_rbs,
            nMBS = args.n_mbs,
            nPBS = args.n_pbs,
            nFBS = args.n_fbs,
            rMBS = args.r_mbs,
            rPBS = args.r_pbs,
            rFBS = args.r_fbs,
            txpowerMBSdBm = args.txpower_mbs_dbm,
            txpowerPBSdBm = args.txpower_pbs_dbm,
            txpowerFBSdBm = args.txpower_fbs_dbm,
            BW = args.bandwidth,
            N0 = args.noise_power,
            QoS_thr = args.qos_thr,
            fc = args.fc,
            x_max = args.x_max,
            y_max = args.y_max,
            Nb = args.nb,
            Nrb = args.nrb_max,
            bsloclist = None,
            prt = False,
        )
        return sce

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs_flat, _ = self.env.reset(seed=seed, options=options)
        
        if self.env.history_channel_information is None:
            self.env.history_channel_information = np.zeros(
                (self.num_agents, self.sce.nUEs, self.sce.nRBs), 
                dtype=np.float32
            )
            dummy_action = np.zeros(sum(self._agent_act_dims))
            self.env.get_obs_4baseline(dummy_action)
        
        obs_list = self._get_obs()
        share_obs_list = self._get_share_obs(obs_list)
        
        obs = np.stack(obs_list, axis=0)[None, ...]
        share_obs = np.stack(share_obs_list, axis=0)[None, ...]
        available_actions = None
        return obs, share_obs, available_actions

    def step(self, actions):
        flat_act = None
        if isinstance(actions, np.ndarray):
            if actions.ndim == 3:
                acts = actions[0]               # (n_agents, max_act_dim)
            elif actions.ndim == 2:
                acts = actions                  # (n_agents, max_act_dim)
            elif actions.ndim == 1:
                # 已经是一维扁平动作，但需要检查是否是真实维度
                flat_act = actions
            else:
                raise ValueError(f"Unexpected action ndim: {actions.ndim}")
        else:
            # list of arrays，先拼成 (n_agents, max_act_dim)
            acts = np.stack(actions, axis=0)

        if flat_act is None:
            # 从 padded actions 中提取真实维度的动作
            real_agent_actions = [
                acts[i, : self._agent_act_dims[i]]
                for i in range(self.num_agents)
            ]
            
            # 分离每个agent的UA和RB动作，然后分别拼接
            all_ua_actions = []
            all_rb_actions = []
            
            for idx, agent_action in enumerate(real_agent_actions):
                # 计算每个agent的UA动作维度
                ua_dim = self._agent_act_dims[idx] // (self.env.sce.nRBs + 1)
                
                # 分离UA和RB动作
                ua_part = agent_action[:ua_dim]
                rb_part = agent_action[ua_dim:]
                
                all_ua_actions.append(ua_part)
                all_rb_actions.append(rb_part)
            
            # [所有UA] + [所有RB]
            flat_act = np.concatenate([
                np.concatenate(all_ua_actions),  # 所有agent的UA动作
                np.concatenate(all_rb_actions)   # 所有agent的RB动作
            ])
            
        
        # print(f"[raw flat_act] min={flat_act.min():.4f}, max={flat_act.max():.4f}")

        # 再做映射
        normed = self._normalize_actions(flat_act)
        # 如果动作不在 [0, 1] 范围内，报错
        if np.any(normed < 0.0) or np.any(normed > 1.0):
            raise ValueError(f"Normalized actions out of bounds: {normed}")
        flat_act = normed
        # todo 映射回0-1再给到reward计算
        
        # def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        #     """
        #     Rescale the action from [-1, 1] to [low, high]
        #     (no need for symmetric action space)

        #     :param scaled_action: Action to un-scale
        #     """
        #     assert isinstance(
        #         self.action_space, spaces.Box
        #     ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
        #     low, high = self.action_space.low, self.action_space.high
        #     return low + (0.5 * (scaled_action + 1.0) * (high - low))
        
        # 问问gpt 如何将1d tensor里的数值映射到给定范围【a,b]之间
        # min-max-normalization； gussian normalization；

        # 检查你这里的计算的数值和 MarlCustomEnv4 中的 MARLstep_withCurrentH 是否一致
        # 注意：这里的 flat_act 需要是一个一维数组，包含所有 agent 的动作
        # 删除那个翻转符号的

        # 验证 flat_act 的维度是否正确
        expected_total_dim = sum(self._agent_act_dims)
        if len(flat_act) != expected_total_dim:
            print(f"Warning: flat_act dimension mismatch. Expected: {expected_total_dim}, Got: {len(flat_act)}")
            print(f"Agent action dims: {self._agent_act_dims}")
            
            # 如果维度不匹配，截断或填充
            if len(flat_act) > expected_total_dim:
                flat_act = flat_act[:expected_total_dim]
            else:
                padded_act = np.zeros(expected_total_dim)
                padded_act[:len(flat_act)] = flat_act
                flat_act = padded_act

        # Optional: Local Search to refine action before stepping
        # Apply only on configured interval to reduce frequency
        if self.use_local_search and (self.ls_interval > 0) and (self.step_count % self.ls_interval == 0):
            try:
                # UA_total length = sum over agents of per-agent UA length
                ua_total = 0
                for idx in range(self.num_agents):
                    ua_dim_i = self._agent_act_dims[idx] // (self.env.sce.nRBs + 1)
                    ua_total += ua_dim_i
                # Choose algorithm
                if self.ls_stepwise:
                    # Stepwise greedy set-to-1, supports UA + RB
                    budget = self.ls_budget if self.ls_budget > 0 else None
                    improved = local_search_stepwise(
                        flat_act,
                        self.env,
                        ua_len=ua_total,
                        include_ua=True,
                        include_rb=self.ls_include_rb,
                        budget=budget,
                        order="ua_then_rb",
                    )
                else:
                    # Legacy stochastic UA-only search
                    if self.ls_budget > 0 and self.ls_k > 0:
                        improved = local_search_improve(
                            flat_act,
                            self.env,
                            ua_len=ua_total,
                            budget=self.ls_budget,
                            k=self.ls_k,
                            early_stop=self.ls_early_stop,
                        )
                    else:
                        improved = flat_act
                # Ensure improved action is in [0,1]
                improved = np.clip(improved, 0.0, 1.0)
                flat_act = improved
            except Exception as e:
                print(f"[LocalSearch] Skipped due to error: {e}")

        # Apply SNR constraints projection: unique UE association, BS capacity Nb, RB cap Nrb
        try:
            flat_act = self._apply_snr_constraints(flat_act)
        except Exception as e:
            print(f"[ConstraintProjection] Skipped due to error: {e}")

        rewards_agent = self._calculate_rewards(flat_act)

        obs_list       = self._get_obs()
        share_obs_list = self._get_share_obs(obs_list)
        obs       = np.stack(obs_list,       axis=0)[None, ...]
        share_obs = np.stack(share_obs_list, axis=0)[None, ...]
        rewards   = rewards_agent.reshape(self.num_agents)[None, :, None]
        self.step_count += 1
        done_flag   = (self.step_count >= self.args.episode_length)
        dones       = np.array([[[done_flag] for _ in range(self.num_agents)]])
        infos       = [{} for _ in range(self.num_agents)]
        available_actions = None

        return obs, share_obs, rewards, dones, infos, available_actions

    def _apply_snr_constraints(self, flat_act: np.ndarray) -> np.ndarray:
        """
        Project action onto feasible set defined by SNR module constraints:
        - Each UE associated to at most one BS (unique association)
        - Each BS associates at most Nb users
        - For associated (b,u), allocate at most Nrb RBs (keep top-Nrb)
        - Non-associated (b,u) must have zero RBs

        Assumes per-agent action layout: [UA(m_i), RB(m_i*nRBs)], and overall
        flat_act is concatenation over agents.
        """
        x = np.asarray(flat_act, dtype=np.float32).ravel().copy()
        n_agents = self.num_agents
        nUEs = self.env.sce.nUEs
        nRBs = self.env.sce.nRBs
        Nb_cap = int(getattr(self.env.sce, 'Nb', 0) or getattr(self.sce, 'Nb', 0) or 0)
        Nrb_cap = int(getattr(self.env.sce, 'Nrb', 0) or getattr(self.sce, 'Nrb', 0) or 0)
        if nRBs <= 0:
            return x

        # Compute per-agent slices
        act_dims = self._agent_act_dims
        offsets = np.cumsum([0] + act_dims[:-1])

        # Map (b, local_u) -> global_u and UA value
        # And build reverse map for unique association resolution
        # Note: BS.UE_set indexes are 1-based in env, convert to 0-based
        bs_list = self.env.BSs
        # Collect structures for later manipulation
        per_agent_info = []  # list of dict: {ua_idx, rb_idx_range, global_u}
        assoc_candidates = {u: [] for u in range(nUEs)}

        for b in range(n_agents):
            start = offsets[b]
            total_dim = act_dims[b]
            if total_dim == 0:
                per_agent_info.append([])
                continue
            # infer m_i (UA length per BS)
            m_i = total_dim // (nRBs + 1)
            ua_start, ua_end = start, start + m_i
            rb_start, rb_end = ua_end, ua_end + m_i * nRBs
            ua_vec = x[ua_start:ua_end]
            # global UE indices
            ue_set = np.array(bs_list[b].UE_set, dtype=int) - 1
            info_list = []
            for j in range(m_i):
                u_global = int(ue_set[j]) if j < len(ue_set) else None
                rb_j_start = rb_start + j * nRBs
                rb_j_end = rb_j_start + nRBs
                info_list.append((ua_start + j, rb_j_start, rb_j_end, u_global))
                if u_global is not None and 0 <= u_global < nUEs:
                    assoc_candidates[u_global].append((b, j, ua_vec[j]))
            per_agent_info.append(info_list)

        # 1) Unique association per UE: keep the BS with max UA value
        keep_mask = { (b,j): False for b in range(n_agents) for j in range(len(per_agent_info[b])) }
        for u, cand in assoc_candidates.items():
            if not cand:
                continue
            # pick max UA
            b_star, j_star, _ = max(cand, key=lambda t: t[2])
            keep_mask[(b_star, j_star)] = True

        # 2) BS capacity: for each BS, among kept candidates, keep top-Nb
        if Nb_cap > 0:
            for b in range(n_agents):
                # gather ua values for kept ones
                kept = []
                for j, (ua_idx, _, _, _) in enumerate(per_agent_info[b]):
                    if keep_mask.get((b, j), False):
                        kept.append((j, x[ua_idx]))
                if len(kept) > Nb_cap:
                    # keep top Nb by ua value
                    kept_sorted = sorted(kept, key=lambda t: t[1], reverse=True)
                    allowed_js = set(j for j, _ in kept_sorted[:Nb_cap])
                    for j, _ in kept:
                        if j not in allowed_js:
                            keep_mask[(b, j)] = False

        # 3) Apply masks to UA and RB, and enforce Nrb cap
        if Nrb_cap < 0:
            Nrb_cap = 0
        for b in range(n_agents):
            for j, (ua_idx, rb_j_start, rb_j_end, u_global) in enumerate(per_agent_info[b]):
                if keep_mask.get((b, j), False):
                    # set UA to 1.0
                    x[ua_idx] = 1.0
                    # RB group: keep top Nrb values, zero others
                    if Nrb_cap > 0:
                        rb_slice = x[rb_j_start:rb_j_end]
                        if rb_slice.size > 0:
                            # indices of top-Nrb
                            topk = np.argsort(-rb_slice)[:min(Nrb_cap, rb_slice.size)]
                            mask = np.zeros_like(rb_slice)
                            mask[topk] = 1.0
                            x[rb_j_start:rb_j_end] = mask
                    else:
                        # no RB allowed, set zero
                        x[rb_j_start:rb_j_end] = 0.0
                else:
                    # not associated => UA 0 and zero RBs
                    x[ua_idx] = 0.0
                    x[rb_j_start:rb_j_end] = 0.0

        return x
    
    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        将动作映射到[0,1]区间
        
        :param actions: 原始动作数组
        :return: 映射到[0,1]区间的动作数组
        """
        return self.map_to_unit_interval_sigmoid(actions)

    def map_to_unit_interval_sigmoid(self, real_values: np.ndarray) -> np.ndarray:
        """
        使用Sigmoid函数将实数映射到(0,1)区间
        
        :param real_values: 任意实数数组
        :return: 映射到(0,1)区间的数组
        """
        # 添加数值稳定性检查
        clipped_values = np.clip(real_values, -500, 500)  # 防止exp溢出
        return 1.0 / (1.0 + np.exp(-clipped_values))

    def map_to_unit_interval_tanh(self, real_values: np.ndarray) -> np.ndarray:
        """
        使用tanh函数将实数映射到[0,1]区间
        
        :param real_values: 任意实数数组
        :return: 映射到[0,1]区间的数组
        """
        return 0.5 * (np.tanh(real_values) + 1.0)
    
    def _get_obs(self) -> list[np.ndarray]:
        """
        返回每个 BS的观测，保持用户维度结构以支持置换等变性。
        返回格式: list of arrays with shape (max_users, nRBs)
        """
        # 确保 history_channel_information 已经过 reset/step 填充
        if self.env.history_channel_information is None:
            obs0, _ = self.env.reset()
            self.env.history_channel_information = obs0.copy()

        # 确保为 numpy array，并按 (BS_num, nUEs, nRBs) 形状
        data = np.array(self.env.history_channel_information, dtype=np.float32)
        if data.ndim != 3:
            data = data.reshape(self.num_agents, self.sce.nUEs, self.sce.nRBs)

        BS_num = self.num_agents
        nUEs = self.env.sce.nUEs
        nRBs = self.env.sce.nRBs

        # 按每个 BS 的 UE_set 拆分
        obs_list = []
        max_users = int(self.max_local_obs_dim // nRBs)
        for b_idx, bs in enumerate(self.env.BSs):
            # UE_set 里是 global UE 索引，从 1 开始减 1
            idx = np.array(bs.UE_set, dtype=int) - 1
            # guard: remove invalid indices
            idx = idx[(idx >= 0) & (idx < nUEs)]
            local = data[b_idx, idx, :].astype(np.float32)      # shape = (len(idx), nRBs)

            # 如果实际用户数超过 max_users，则截断（保持前面用户）
            if local.shape[0] > max_users:
                local = local[:max_users, :]

            # Pad用户维度到最大用户数
            if local.shape[0] < max_users:
                padded = np.zeros((max_users, nRBs), dtype=np.float32)
                padded[:local.shape[0], :] = local
                obs_list.append(padded)
            else:
                obs_list.append(local)

        return obs_list

    def get_agent_order_by_user_count(self) -> list:
        """返回按照每个 BS 关联用户数降序排列的 agent 索引列表。

        说明：该方法仅返回排序索引，是否在训练/优化中按此顺序使用由上层控制器决定。
        """
        ue_counts = [len(bs.UE_set) for bs in self.env.BSs]
        order = sorted(range(self.num_agents), key=lambda i: ue_counts[i], reverse=True)
        return order

    def _get_share_obs(self, obs_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        这里直接广播每个 agent 能看到全部 BS 的 channel info：
        share_obs_list[i] = 全量 channel information (BS_num, nUEs, nRBs)
        """
        share_obs_data = np.array(self.env.history_channel_information).copy()
        # Ensure shape is (num_agents, nUEs, nRBs)
        if share_obs_data.ndim != 3:
            share_obs_data = share_obs_data.reshape(self.num_agents, self.sce.nUEs, self.sce.nRBs)
        return [share_obs_data for _ in range(self.num_agents)]
    
    def _calculate_rewards(self, flat_act: np.ndarray) -> np.ndarray:
        obs, reward, terminated, truncated, info = self.env.MARLstep_withCurrentH(flat_act)        
        return np.full(self.num_agents, reward, dtype=np.float32)
            

    
    def seed(self, seed=None):
        np.random.seed(seed)
    
    def close(self):
        pass