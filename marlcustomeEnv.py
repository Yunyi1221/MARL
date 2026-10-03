#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：UARL 
@File    ：marlcustomeEnv.py
@Author  ：Jiansheng LI
@Date    ：2023/12/31 0:52 
'''

from typing import Any
import copy
import gymnasium as gym
import numpy as np
import torch as th

from RRM import Environment
# from module.rl_utils import load_env, DotDic


class MarlCustomEnv(Environment):
    def __init__(self, sce):

        super().__init__(sce)
        # TODO check Encoder
        self.history_channel_information = None
        self.dtype = np.float32
        # self.enc1 = OneHotEncoder(sparse=False)
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        # self.TxPowerVector = np.array([b.Transmit_Power_dBm() for b in self.BSs])
        # b, k, u = self.BS_num, sce.nRBs, sce.nUEs
        # self.MatrixE = np.zeros((u, k * u))

        # for every basestation, its user candidate are the set of user locating in its signal coverage zone.
        # thus, the action space of every agent is the box with dimension of number of users in its signal coverage zone.
        self.action_spaces = []
        self.observation_spaces = []

        totalActionDimension = 0
        for b in self.BSs:
            totalActionDimension += len(b.UE_set)
            self.action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set),)))
            self.observation_spaces.append(
                gym.spaces.box.Box(np.array([-np.inf] * len(b.UE_set)), np.array([np.inf] * len(b.UE_set)),
                                   (len(b.UE_set),),
                                   dtype=self.dtype))
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))

        # if we change the dimension of observation_space,
        # the item list needed to be modified is as follows:
        # self.observation_space(s)
        # distribution_obs() distribution_actions()
        # architecture of mixer

        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(len(self.BSs), len(self.UEs)),
                                                    dtype=self.dtype)
        # prepare onehot encoder for every RB action
        # all_bs_categories = []
        # for b in self.BSs:
        #     all_bs_categories.append(np.arange(len(b.UE_set)))
        # from sklearn.preprocessing import OneHotEncoder
        # self.encset = []
        #
        # for i in all_bs_categories:
        #     self.encset.append(OneHotEncoder(sparse=False).fit(i.reshape(-1, 1)))

        # tmp2=copy.deepcopy(tmp)
        # tmp2.append(gym.spaces.box.Box(low=-np.inf,high=np.inf,shape=(1,)))
        # self.observation_space = gym.spaces.tuple.Tuple(tmp)

        # tmp,max=[],0
        # for b in self.BSs:
        #     if (len(b.UE_set)>max):
        #         max=len(b.UE_set)
        # tmp=[[[max] * self.sce.nRBs]*self.BS_num]
        # self.action_space = gym.spaces.multi_discrete.MultiDiscrete(tmp)

        # self.observation_space = spaces.Dict(
        #     spaces={
        #         "last-action": gym.spaces.multi_discrete.MultiDiscrete(tmp),
        #         "img": spaces.Box(-np.inf, np.inf,shape=(1,)),
        #     }
        # )
        # t1 = self.sce.nRBs * self.BS_num
        # t2 = self.history_channel_information.shape[0]
        # t2 = t1
        # todo define t2
        # lower_bound = [0] * t1
        #
        # lower_bound.extend([-np.inf] * t2)
        #
        # upper_bound = tmp
        # # [self.sce.nUEs] * t1
        # upper_bound.extend([np.inf] * t2)

        # self.observation_space = gym.spaces.box.Box(np.array(lower_bound), np.array(upper_bound), (t1 + t2,),
        #                                             dtype=np.float64)
        # self.history_channel_information = np.zeros((self.sce.nRBs, self.BS_num))
        self.MaxReward = 0
        self.StopThreshold = 0.05
        self.LastReward = 0
        self.LastAction = None

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def Is_vaild_action(self, actions):
        for b_index, b in enumerate(self.BSs):
            rb_assignment_action = actions[b_index * self.sce.nRBs:(b_index + 1) * self.sce.nRBs]
            if len(set(rb_assignment_action)) < len(rb_assignment_action):
                return False
        return True

    def compute_cost(self, actions: np.ndarray):
        actions = actions.reshape(self.BS_num, self.sce.nRBs, 1)
        cost1 = 0
        # for i in range(self.BS_num):
        #     step2 = self.encset[i].fit_transform(actions[i]).sum(axis=0)
        #     cost1+= step2.sum(axis=0) - len(step2)
        for i in range(self.BS_num):
            # 获取唯一元素和它们的计数
            elements, counts = np.unique(actions[i], return_counts=True)
            # 设置要忽略的元素
            ignore_element = len(self.BSs[i].UE_set) + 1
            # 计算重复元素的数量，忽略 ignore_element
            # cost1 += np.sum(counts[elements != ignore_element]-1)
            cost1 += np.sum(counts[elements != ignore_element] - 1)
            # print("忽略元素 {} 后的重复元素数量：".format(ignore_element), num_duplicates)
        # = (np.apply_along_axis(self.func_cost1, arr=actions.reshape(self.sce.nBSs, self.sce.nRBs), axis=1)
        #          .sum(axis=0))
        return cost1

    def compute_cost_stdout(self, actions: np.ndarray):
        actions = actions.reshape(self.BS_num, self.sce.nRBs, 1)
        cost1 = 0
        # for i in range(self.BS_num):
        #     step2 = self.encset[i].fit_transform(actions[i]).sum(axis=0)
        #     cost1+= step2.sum(axis=0) - len(step2)
        for i in range(self.BS_num):
            # 获取唯一元素和它们的计数
            elements, counts = np.unique(actions[i], return_counts=True)
            # 设置要忽略的元素
            ignore_element = len(self.BSs[i].UE_set) + 1
            # 计算重复元素的数量，忽略 ignore_element
            # cost1 += np.sum(counts[elements != ignore_element]-1)

            cost1 += np.sum(counts[elements != ignore_element] - 1)
            print(f'BS{i}: elements: {elements}, counts: {counts - 1}')
        print()
        print('total cost: ', cost1)
        # print("忽略元素 {} 后的重复元素数量：".format(ignore_element), num_duplicates)
        # = (np.apply_along_axis(self.func_cost1, arr=actions.reshape(self.sce.nBSs, self.sce.nRBs), axis=1)
        #          .sum(axis=0))

    def step_with_cost(self, actions):
        obs, reward, terminated, truncated, info = self.Pathloss_3GPP_RLStep_HighPerformance(actions)
        info['cost'] = self.compute_cost(actions)
        return obs, reward, terminated, truncated, info

    # def func_H_power(self, local_ue_index, b_index):
    #     b = self.BSs[b_index]
    #     global_u_index = b.UE_set[local_ue_index]
    #     distance = self.distance_matrix
    #     signal_power, channel_power \
    #         = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
    #     return signal_power, channel_power

    def Pathloss_3GPP_RLStep_HighPerformance(self, actions):
        actions = actions.reshape(self.BS_num, self.sce.nRBs)
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
        total_rate = 0
        signal_power_set = np.zeros((self.BS_num, self.sce.nRBs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.sce.nRBs, self.BS_num))

        for b_index, b in enumerate(self.BSs):
            rb_assignment_action = actions[b_index]
            for rb_index, u in enumerate(rb_assignment_action):
                if u == len(b.UE_set):
                    # if Ture, it is a empty assignment,
                    # which means this assignment is not assignment any RB to the user.
                    continue
                global_u_index = b.UE_set[u]
                # Loc_diff = b.Get_Location() - self.UEs[global_u_index].Get_Location()
                # distance = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
                signal_power, channel_power \
                    = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                signal_power_set[b_index][rb_index] = signal_power
                channal_power_set[b_index][rb_index] = channel_power
        # total_rate2=0
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nRBs)

        interference_m = interference_sum_m - signal_power_set + Noise + 1e-7
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        obs = np.concatenate((actions.reshape(-1, ), channal_power_set.reshape(-1, )), axis=0)

        reward = total_rate
        self.history_channel_information = channal_power_set
        terminated, truncated, info = False, False, {}

        # if abs(reward - self.LastReward) < self.StopThreshold:
        #     terminated = True

        self.LastReward = reward
        self.MaxReward = max(reward, self.MaxReward)

        return obs, reward, terminated, truncated, info

    def Pathloss_3GPP_RLStep(self, actions):
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
        total_rate = 0
        rb_interference_set = np.zeros((self.BS_num, self.sce.nRBs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        history_channel_information = np.zeros((self.sce.nRBs, self.BS_num))

        for b_index, b in enumerate(self.BSs):
            rb_assignment_action = actions[b_index * self.sce.nRBs:(b_index + 1) * self.sce.nRBs]
            for rb_index, u in enumerate(rb_assignment_action):
                if u == len(b.UE_set):
                    continue
                global_u_index = b.UE_set[u]
                # Loc_diff = b.Get_Location() - self.UEs[global_u_index].Get_Location()
                # distance = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
                signal_power, channel_power \
                    = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                rb_interference_set[b_index][rb_index] = signal_power
                history_channel_information[b_index][rb_index] = channel_power
        # total_rate2=0
        for k_set in rb_interference_set:
            interference_sum = k_set.sum(axis=0)
            for i in k_set:
                interference = interference_sum - i
                SINR = i / (interference + Noise + 1e-7)
                un_scale_Rate = np.log2(1 + SINR)
                total_rate += un_scale_Rate
                # Rate = self.sce.BW * np.log2(1 + SINR) / (10 ** 6)
                # total_rate2 += Rate
        total_rate = self.sce.BW * total_rate / (10 ** 6)
        # assert total_rate==total_rate2
        # np.array(history_channel_information)
        obs = np.concatenate((actions, history_channel_information.reshape(-1, )), axis=0)
        reward = total_rate
        self.history_channel_information = history_channel_information
        terminated, truncated, info = False, False, {}

        # if abs(reward - self.LastReward) < self.StopThreshold:
        #     terminated = True

        self.LastReward = reward
        self.MaxReward = max(reward, self.MaxReward)

        return obs, reward, terminated, truncated, info

    def RLstep(self, actions):
        # if not self.Is_vaild_action(actions):
        #     reward = -1
        #     terminated, truncated, info = False, False, {}
        #     obs = np.concatenate((actions,self.history_channel_information ), axis=0)
        #
        #     return np.array(obs), reward, terminated, truncated, info

        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
        total_rate = 0
        rb_interference_set = [[] for i in
                               range(self.sce.nRBs)]  # store all users' the signal power in every subcarrier
        history_channel_information = []

        for b_index, b in enumerate(self.BSs):
            rb_assignment_action = actions[b_index * self.sce.nRBs:(b_index + 1) * self.sce.nRBs]
            for rb_index, u in enumerate(rb_assignment_action):
                global_u_index = b.UE_set[u]

                Loc_diff = b.Get_Location() - self.UEs[global_u_index].Get_Location()
                distance = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

                signal_power, channel_power = self.cal_Receive_Power(b, distance)

                rb_interference_set[rb_index].append(signal_power)
                history_channel_information.append(channel_power)

        for k_set in rb_interference_set:
            interference_sum = sum(k_set)
            for i in k_set:
                interference = interference_sum - i
                SINR = i / (interference + Noise + 1e-7)
                Rate = self.sce.BW * np.log2(1 + SINR) / (10 ** 6)
                total_rate += Rate

        self.history_channel_information = np.array(history_channel_information)
        obs = np.concatenate((actions, history_channel_information), axis=0)
        reward = total_rate

        terminated, truncated, info = False, False, {}

        #
        # if abs(reward - self.LastReward) < self.StopThreshold:
        #     terminated = True

        self.LastReward = reward
        self.MaxReward = max(reward, self.MaxReward)

        return obs, reward, terminated, truncated, info

        # return five values: obs, reward, terminated, truncated, info

    def RLreset(self, seed=None, options=None):
        # self.history_channel_information = np.array([0] * (self.history_channel_information.shape[0]))
        init_action = np.zeros((self.action_space.shape[0],))
        obs = np.concatenate((init_action, self.history_channel_information.reshape(-1, )), axis=0)
        # self.history_channel_information = np.zeros((self.sce.nBSs,self.sce.nRBs))
        # obs = [0] * (self.action_space.shape[0] + self.history_channel_information.shape[0])
        observation, info = np.array(obs), {}
        self.MaxReward = 0
        self.StopThreshold = 1e-3
        self.LastReward = 0
        return observation, info

    def MARLreset(self, seed=None, options=None):
        # raise NotImplementedError
        self.LastAction = th.randn(self.action_space.shape[0]) * 1e-6
        # obs = np.ones(shape=(len(self.BSs), len(self.UEs)), dtype=self.dtype)
        # obs, _, _, _, _ = self.MARLstep(th.tensor(self.action_space.sample()))
        # act = []
        # for i in self.action_spaces:
        #     act.append(i.sample())
        # act = np.concatenate(act)
        # obs, _, _, _, _ = self.MARLstep(act)
        obs = self.observation_space.sample()
        # self.history_channel_information = np.zeros((self.sce.nBSs,self.sce.nRBs))
        # obs = [0] * (self.action_space.shape[0] + self.history_channel_information.shape[0])
        observation, info = np.array(obs), {}
        self.StopThreshold = 1e-3
        # self.LastAction= np.zeros((self.action_space.shape[0],))
        return observation, info

    def terminate_condtion(self, actions: th.tensor):
        # actions: 1d - torch.tensor

        return bool(th.sum(th.kl_div(th.log(actions), self.LastAction)) <= self.StopThreshold)
        # re=np.

    # def MARLstep(self, actions_total: np.ndarray) -> tuple[Any, float, bool, bool, dict[Any, Any]]:
    #     """
    #     MARL step function
    #     :param actions_total: shape(len(sum all the b.UE_set))
    #     :return:
    #     - observation_total: shape(len(self.BSs)*len(self.UEs);
    #     - reward: one reward for all agents;
    #     - terminated: one terminated for all agents
    #     """
    #
    #     # def distribute_actions(self, actions: Tensor) -> list:
    #     #     # obs shape: (len(self.BSs), len(self.UEs))
    #     #     # 我们要根据基站的信号范围内的用户集合从总的obs中提取并用来分发给每个智能体的obs
    #     #     new_actions = []
    #     #     try:
    #     #         index = 0
    #     #         for i, b in enumerate(self.original_env.BSs):
    #     #             new_actions.append(actions[:, index:index + len(b.UE_set)])
    #     #             index += len(b.UE_set)
    #     #     except Exception as e:
    #     #         print("actions_total do not match the determined distribution rule", str(NotImplementedError), "\n",
    #     #               str(e))
    #     #
    #     #     return new_actions
    #
    #     # if we change the dimension of observation_space,
    #     # the item list needed to be modified is as follows:
    #     # self.observation_space(s)
    #     # distribution_obs() distribution_actions()
    #     # architecture of mixer
    #
    #     bs_wise_action = []
    #     index = 0
    #     for i, b in enumerate(self.BSs):
    #         bs_wise_action.append(actions_total[index:index + len(b.UE_set)])
    #         index += len(b.UE_set)
    #     # .reshape(self.sce.nBSs, self.sce.nRBs)
    #     Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
    #
    #     signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
    #     # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
    #     channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
    #     # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
    #     # self.sce.nBSs)]
    #     a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理
    #
    #     for b_index, b in enumerate(self.BSs):
    #         s_assignment_bs = bs_wise_action[b_index]
    #         for local_u_index, s_b_u in enumerate(s_assignment_bs):
    #             global_u_index = b.UE_set[local_u_index] - 1
    #             for rb_index in range(self.sce.nRBs):
    #                 signal_power, channel_power \
    #                     = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
    #                 # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
    #                 signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * signal_power
    #                 channal_power_set[b_index][global_u_index][rb_index] = s_b_u * a_b_k_u * channel_power
    #
    #         # for rb_index, u in enumerate(rb_assignment_action):
    #         #     if u == len(b.UE_set):
    #         #         # if Ture, it is a empty assignment,
    #         #         # which means this assignment is not assignment any RB to the user.
    #         #         continue
    #         #     global_u_index = b.UE_set[u]
    #         #     # Loc_diff = b.Get_Location() - self.UEs[global_u_index].Get_Location()
    #         #     # distance = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
    #         #     signal_power, channel_power \
    #         #         = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
    #         #     signal_power_set[b_index][rb_index] = signal_power
    #         #     channal_power_set[b_index][rb_index] = channel_power
    #     interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
    #     # interference_sum = interference_sum.reshape(-1,1)
    #     interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
    #
    #     interference_m = interference_sum_m - signal_power_set + Noise + 1e-7
    #     unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
    #     total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)
    #
    #     obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
    #
    #     reward = total_rate
    #
    #     self.history_channel_information = obs
    #     if not isinstance(actions_total, th.Tensor):
    #         actions_total = th.tensor(actions_total)
    #     terminated, truncated, info = self.terminate_condtion(actions_total), False, {}
    #     self.LastAction = actions_total
    #
    #     # if abs(reward - self.LastReward) < self.StopThreshold:
    #     #     terminated = True
    #
    #     # self.LastReward = reward
    #     # self.MaxReward = max(reward, self.MaxReward)
    #
    #     return obs, reward, terminated, truncated, info
    #
    #     # raise NotImplementedError
    #     # obs = np.ones(shape=(len(self.BSs),len(self.UEs)))
    #
    #     # obs = []
    #     # for i, _ in enumerate(self.BSs):
    #     #     obs.append(self.observation_spaces[i].sample())
    #     # obs = np.concatenate(obs)
    #
    #     # reward = 0.0
    #     # # terminateds, truncateds, infos = [False]*len(self.BSs), [False]*len(self.BSs), [{}]*len(self.BSs)
    #     # terminateds, truncateds, infos = False, False, {}
    #     #
    #     # return obs, reward, terminateds, truncateds, infos
    def MARLstep(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:

        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """
        # reward, terminated, truncated, info = 1, False, False, {'actions': actions_user_association_RBG_assignment}
        # obs = self.observation_space.sample()
        # return obs, reward, terminated, truncated, info
        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.

        # bs_wise_RBG_assignment_action = []
        # actions_RBG_assignment_total = actions_user_association_RBG_assignment[1]
        # index = 0
        # for i, b in enumerate(self.BSs):
        #     stride = len(b.UE_set)*self.sce.nRBs
        #     bs_wise_RBG_assignment_action.append(actions_RBG_assignment_total[:, index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
        #     index += stride

        bs_wise_user_association_action = actions_user_association_RBG_assignment[0]
        bs_wise_RBG_assignment_action = actions_user_association_RBG_assignment[1]
        # .reshape(self.sce.nBSs, self.sce.nRBs)
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise

        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * signal_power
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power

        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise + 1e-7
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs
        # if not isinstance(actions_RBG_assignment_total, th.Tensor):
        #     actions_RBG_assignment_total = th.tensor(actions_RBG_assignment_total)
        # terminated, truncated, info = self.terminate_condtion(actions_RBG_assignment_total), False, {}
        # self.LastAction = actions_RBG_assignment_total
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info

    def step(self, actions):
        # return self.RLstep(actions)
        # return self.Pathloss_3GPP_RLStep(actions)
        # return self.step_with_cost(actions)
        return self.MARLstep(actions)

    def reset(self, seed=None, options=None):
        # observation, info = self.RLreset(seed, options)
        observation, info = self.MARLreset(seed, options)

        return observation, info


class MarlCustomEnv2(MarlCustomEnv):
    # for user association and RBG assignment environment

    # if we change the dimension of observation_space,
    # the item list needed to be modified is as follows:
    # self.observation_space(s)
    # distribution_obs() distribution_actions()
    # architecture of mixer

    # todo check whether run the __init__() when loading pickle file of env
    def __init__(self, sce, opt, env_instance: MarlCustomEnv = None):
        if env_instance is None:
            super(MarlCustomEnv, self).__init__(sce, opt)
        else:
            self.__setstate__(env_instance.__getstate__())
        self.action_spaces = []
        self.observation_spaces = []
        totalActionDimension = 0
        for b in self.BSs:
            totalActionDimension += len(b.UE_set) * self.sce.nRBs
            self.action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) * self.sce.nRBs,)))
            self.observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf,
                                   (len(b.UE_set) * self.sce.nRBs,),
                                   dtype=self.dtype)
            )
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf,
                                                    shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
                                                    dtype=self.dtype)

    def MARLstep(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:

        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """
        # reward, terminated, truncated, info = 1, False, False, {'actions': actions_user_association_RBG_assignment}
        # obs = self.observation_space.sample()
        # return obs, reward, terminated, truncated, info
        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[1]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[:, index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        bs_wise_user_association_action = actions_user_association_RBG_assignment[0]
        # .reshape(self.sce.nBSs, self.sce.nRBs)
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise

        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * signal_power
                    channal_power_set[b_index][global_u_index][rb_index] = s_b_u * a_b_k_u * channel_power

        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise + 1e-7
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs
        # if not isinstance(actions_RBG_assignment_total, th.Tensor):
        #     actions_RBG_assignment_total = th.tensor(actions_RBG_assignment_total)
        # terminated, truncated, info = self.terminate_condtion(actions_RBG_assignment_total), False, {}
        # self.LastAction = actions_RBG_assignment_total
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info

    def MARLreset(self, seed=None, options=None):
        self.LastAction = th.randn(self.action_space.shape[0]) * 1e-6
        # obs = np.ones(shape=(len(self.BSs), len(self.UEs)), dtype=self.dtype)
        # obs, _, _, _, _ = self.MARLstep(th.tensor(self.action_space.sample()))
        # self.history_channel_information = np.zeros((self.sce.nBSs,self.sce.nRBs))
        # obs = [0] * (self.action_space.shape[0] + self.history_channel_information.shape[0])
        obs = self.observation_space.sample()
        observation, info = np.array(obs), {}
        self.MaxReward = 0
        self.StopThreshold = 1e-3
        self.LastReward = 0
        # self.LastAction= np.zeros((self.action_space.shape[0],))
        return observation, info


class MarlCustomEnv3(MarlCustomEnv):
    def __init__(self, sce, env_instance: MarlCustomEnv = None):
        if env_instance is None:
            super(MarlCustomEnv, self).__init__(sce)
        else:
            self.__setstate__(env_instance.__getstate__())
        # TODO check Encoder
        self.history_channel_information = None
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        # self.TxPowerVector = np.array([b.Transmit_Power_dBm() for b in self.BSs])
        # b, k, u = self.BS_num, sce.nRBs, sce.nUEs
        # self.MatrixE = np.zeros((u, k * u))
        # for every basestation, its user candidate are the set of user locating in its signal coverage zone.
        # thus, the action space of every agent is the box with dimension of number of users in its signal coverage zone.

        self.user_association_action_spaces = []
        self.rbg_assignment_action_spaces = []
        self.user_association_observation_spaces = []
        self.rbg_assignment_observation_spaces = []
        # totalActionDimension_ua=0
        # totalActionDimension_rbg=0
        totalActionDimension = 0

        for b in self.BSs:
            # for actor policy input parameter
            totalActionDimension += len(b.UE_set) + len(b.UE_set) * self.sce.nRBs
            # totalActionDimension_ua += len(b.UE_set)
            self.user_association_action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set),)))
            self.user_association_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (len(b.UE_set),), dtype=self.dtype))

            # totalActionDimension_rbg += len(b.UE_set) * self.sce.nRBs
            self.rbg_assignment_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) * self.sce.nRBs,)))
            self.rbg_assignment_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf,
                                   (len(b.UE_set) * self.sce.nRBs,),
                                   dtype=self.dtype)
            )

        # for replay buffer
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf,
                                                    shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
                                                    dtype=self.dtype)

        # self.action_space_ua = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension_ua,))
        # self.observation_space_ua = gym.spaces.box.Box(low=-np.inf, high=np.inf,
        #                                             shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
        #                                             dtype=self.dtype)
        #
        # self.action_space_rbg = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension_rbg,))
        # self.observation_space_rbg = gym.spaces.box.Box(low=-np.inf, high=np.inf,
        #                                             shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
        #                                             dtype=self.dtype)

        # if we change the dimension of observation_space,
        # the item list needed to be modified is as follows:
        # self.observation_space(s)
        # distribution_obs() distribution_actions()
        # self.predict4()
        # for critic input parameter
        self.share_critic_action_spaces = []
        self.share_critic_observation_spaces = []

        for b in self.BSs:
            self.share_critic_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) + len(b.UE_set) * self.sce.nRBs,)))
            self.share_critic_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (len(b.UE_set) * self.sce.nRBs,), dtype=self.dtype))

        self.StopThreshold = 0.05
        self.LastAction = None

    def MARLstep(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:

        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """
        # reward, terminated, truncated, info = 1, False, False, {'actions': actions_user_association_RBG_assignment}
        # obs = self.observation_space.sample()
        # return obs, reward, terminated, truncated, info
        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        # .reshape(self.sce.nBSs, self.sce.nRBs)
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise

        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * signal_power
                    channal_power_set[b_index][global_u_index][rb_index] = s_b_u * a_b_k_u * channel_power

        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise + 1e-7
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs
        # if not isinstance(actions_RBG_assignment_total, th.Tensor):
        #     actions_RBG_assignment_total = th.tensor(actions_RBG_assignment_total)
        # terminated, truncated, info = self.terminate_condtion(actions_RBG_assignment_total), False, {}
        # self.LastAction = actions_RBG_assignment_total
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info

    def MARLreset(self, seed=None, options=None):
        self.LastAction = th.randn(self.action_space.shape[0]) * 1e-6
        # obs = np.ones(shape=(len(self.BSs), len(self.UEs)), dtype=self.dtype)
        # obs, _, _, _, _ = self.MARLstep(th.tensor(self.action_space.sample()))
        # self.history_channel_information = np.zeros((self.sce.nBSs,self.sce.nRBs))
        # obs = [0] * (self.action_space.shape[0] + self.history_channel_information.shape[0])
        obs = self.observation_space.sample()
        observation, info = np.array(obs), {}
        self.MaxReward = 0
        self.StopThreshold = 1e-3
        self.LastReward = 0
        # self.LastAction= np.zeros((self.action_space.shape[0],))
        return observation, info


# for MADDPG+VDN mixer, the every individual critic use the local observation and observation of neighbor agents

class MarlCustomEnv4(MarlCustomEnv):
    # change actionspace and obs space
    # change MARLsetp logits
    # before this version, the marlstep use the next time slot obs to cal the current action's reward which is weired
    # now, we store the obs in the self.history_channel_information then use it to calculate the reward
    # the new function has the notation called MARLstep_withCurrentH
    # you can search the word : MARLstep_withCurrentH
    def __init__(self, sce, env_instance: MarlCustomEnv = None):

        self.flag = True
        if env_instance is None:
            super().__init__(sce)
        else:
            self.__setstate__(env_instance.__getstate__())
            self.sce = sce
        # TODO check Encoder
        self.history_channel_information = None
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        self.use_action_projection = False
        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        # self.TxPowerVector = np.array([b.Transmit_Power_dBm() for b in self.BSs])
        # b, k, u = self.BS_num, sce.nRBs, sce.nUEs
        # self.MatrixE = np.zeros((u, k * u))
        # for every basestation, its user candidate are the set of user locating in its signal coverage zone.
        # thus, the action space of every agent is the box with dimension of number of users in its signal coverage zone.

        self.user_association_action_spaces = []
        self.rbg_assignment_action_spaces = []
        self.user_association_observation_spaces = []
        self.rbg_assignment_observation_spaces = []
        # totalActionDimension_ua=0
        # totalActionDimension_rbg=0
        totalActionDimension = 0

        for b in self.BSs:
            # for actor policy input parameter
            totalActionDimension += len(b.UE_set) + len(b.UE_set) * self.sce.nRBs
            # totalActionDimension_ua += len(b.UE_set)
            self.user_association_action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set),)))
            self.user_association_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (len(b.UE_set),), dtype=self.dtype))

            # totalActionDimension_rbg += len(b.UE_set) * self.sce.nRBs
            self.rbg_assignment_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) * self.sce.nRBs,)))
            self.rbg_assignment_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf,
                                   (len(b.UE_set) * self.sce.nRBs,),
                                   dtype=self.dtype)
            )

        # for replay buffer
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf,
                                                    shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
                                                    dtype=self.dtype)
        self.action_spaces=[]
        for b in self.BSs:
            # for actor policy input parameter
            dim = len(b.UE_set) + len(b.UE_set) * self.sce.nRBs
            # totalActionDimension_ua += len(b.UE_set)
            self.action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(dim,)))
            self.observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (len(b.UE_set)*self.sce.nRBs,), dtype=self.dtype))

            
        # self.action_space_ua = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension_ua,))
        # self.observation_space_ua = gym.spaces.box.Box(low=-np.inf, high=np.inf,
        #                                             shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
        #                                             dtype=self.dtype)
        #
        # self.action_space_rbg = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension_rbg,))
        # self.observation_space_rbg = gym.spaces.box.Box(low=-np.inf, high=np.inf,
        #                                             shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
        #                                             dtype=self.dtype)

        # if we change the dimension of observation_space,
        # the item list needed to be modified is as follows:
        # self.observation_space(s)
        # distribution_obs() distribution_actions()
        # self.predict4()
        # for critic input parameter
        self.share_critic_action_spaces = []
        self.share_critic_observation_spaces = []

        for b in self.BSs:
            self.share_critic_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) + len(b.UE_set) * self.sce.nRBs,)))
            self.share_critic_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (totalActionDimension // (self.sce.nRBs + 1) * self.sce.nRBs,),
                                   dtype=self.dtype))

        self.StopThreshold = 0.05
        self.LastAction = None

    def distribute_actions(self, actions, mode=0) -> list:
        """
        :param actions: actions_total
        :return: [[agent1 ua action, agent2 ua action, ...] [agent1 rbg action, agent 2 rbg action, ...]]
        """
        actions_user_association_RBG_assignment = actions
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        # only for facmac5
        # bs_wise_RBG_assignment_action = []
        # bs_wise_user_association_action = []
        # index = 0
        # for i, b in enumerate(self.BSs):
        #     stride = len(b.UE_set)
        #     bs_wise_user_association_action.append(
        #         actions_user_association_RBG_assignment[..., index:index + stride])
        #     bs_wise_RBG_assignment_action.append(
        #         actions_user_association_RBG_assignment[..., index + stride:index + stride*(self.sce.nRBs+1)].reshape(-1,stride,self.sce.nRBs))
        #     index += stride*(self.sce.nRBs+1)
        # .reshape(self.sce.nBSs, self.sce.nRBs)
        return [bs_wise_user_association_action, bs_wise_RBG_assignment_action]

    def __setstate__(self, state):
        self.__dict__.update(state)
        # for ue in self.UEs:
        #     for b in self.BSs:
        #         if ue.id in b.UE_set:
        #             ue.insideBSregion=[]
    def get_obs_4baseline(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:
        # todo action是基于obs(t)给出的 输入到step里后 输出的reward 是基于 obs(t+1)算的，这是否合理
        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """
        # reward, terminated, truncated, info = 1, False, False, {'actions': actions_user_association_RBG_assignment}
        # obs = self.observation_space.sample()
        # return obs, reward, terminated, truncated, info
        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        # .reshape(self.sce.nBSs, self.sce.nRBs)
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise

        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power_dB \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    # signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * signal_power
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power_dB
        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        self.history_channel_information = obs
        return obs

    def MARLstep(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:
        return self.MARLstep_withCurrentH(actions_user_association_RBG_assignment)

    def project_rbg_action(self, bs_wise_actions, threshold=0.5):
        if self.sce.rbg_N_b is None:
            self.sce.rbg_N_b = 3
        N_rb = self.sce.rbg_N_b
        action_rbg_bs_wise_proj = []
        for tensor in bs_wise_actions:
            if not isinstance(tensor, th.Tensor):
                tensor = th.tensor(tensor)
            if threshold is None:
                _, indices = th.topk(tensor, N_rb, dim=-1)
                new_tensor = th.zeros_like(tensor)
                new_tensor.scatter_(-1, indices, 1)
                action_rbg_bs_wise_proj.append(new_tensor)
            else:
                new_tensor = th.zeros_like(tensor)
                _, topk_indices = th.topk(tensor, N_rb, dim=-1)
                new_tensor.scatter_(-1, topk_indices, 1)
                new_tensor[tensor < threshold] = 0
                action_rbg_bs_wise_proj.append(new_tensor)
        return action_rbg_bs_wise_proj

    def MARLstep_withCurrentH(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:
        # current step function
        # todo action是基于obs(t)给出的 输入到step里后 输出的reward 是基于 obs(t+1)算的，这是否合理
        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """

        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        # for not FACMAC5
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        if self.use_action_projection:
            bs_wise_RBG_assignment_action = self.project_rbg_action(bs_wise_RBG_assignment_action)
        # only for facmac5
        # bs_wise_RBG_assignment_action = []
        # bs_wise_user_association_action = []
        # index = 0
        # for i, b in enumerate(self.BSs):
        #     stride = len(b.UE_set)
        #     bs_wise_user_association_action.append(
        #         actions_user_association_RBG_assignment[..., index:index + stride])
        #     bs_wise_RBG_assignment_action.append(
        #         actions_user_association_RBG_assignment[..., index + stride:index + stride*(self.sce.nRBs+1)].reshape(-1,stride,self.sce.nRBs))
        #     index += stride*(self.sce.nRBs+1)
        # .reshape(self.sce.nBSs, self.sce.nRBs)
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理
        assert self.history_channel_information is not None
        # if self.history_channel_information is not None:
        H_dB = self.history_channel_information.reshape((self.BS_num, self.sce.nUEs, self.sce.nRBs), )
        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index].squeeze()
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]

                    _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    # s_b_u *
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[b_index, global_u_index, rb_index] / 10))
                    # print(f'sbku before:{signal_power_set[rb_index][global_u_index]}\nafter{signal_power_set[rb_index][global_u_index]*s_b_u}')
                    # H[b_index,global_u_index,rb_index] * b.Transmit_Power_dBm()
                    # 注意 H_dB 是 fading - pathloss
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm
        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise
        if np.any(1 + signal_power_set / interference_m <= 0):
            print("Warning: log2 input is non-positive")
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs
        # if not isinstance(actions_RBG_assignment_total, th.Tensor):
        #     actions_RBG_assignment_total = th.tensor(actions_RBG_assignment_total)
        # terminated, truncated, info = self.terminate_condtion(actions_RBG_assignment_total), False, {}
        # self.LastAction = actions_RBG_assignment_total
        terminated, truncated, info = False, False, {}

        # 用户随机游走变换位置
        # self.random_walk()
        # 重新计算用户基站距离矩阵
        # for b_index, b in enumerate(self.BSs):
        #     for ue_index, ue in enumerate(self.UEs):
        #         Loc_diff = b.Get_Location() - ue.Get_Location()
        #         self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        return obs, reward, terminated, truncated, info

    def calculate_sumrate_userwise(self, actions_user_association_RBG_assignment, obs):
        """
        Calculate_sumrate based on the input action and obs
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
            obs: channel power matrix
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        H_dB = obs.reshape((self.BS_num, self.sce.nUEs, self.sce.nRBs), )
        UE_rate_list = [[] for _ in range(len(self.UEs))]
        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index].squeeze()
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[b_index, global_u_index, rb_index] / 10))

        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
        interference_m = interference_sum_m - signal_power_set + Noise
        log2input = 1 + signal_power_set / interference_m
        log2input[log2input <= np.zeros_like(log2input)] = 1
        unscale_rate_m = np.log2(log2input)
        # unscale_rate_m = np.log2(1 + 3)
        UE_rate_list = self.sce.BW * unscale_rate_m / (10 ** 6)
        return UE_rate_list

    def calculate_sumrate(self, actions_user_association_RBG_assignment, obs):
        """
        Calculate_sumrate based on the input action and obs
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
            obs: channel power matrix
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """

        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        H_dB = obs.reshape((self.BS_num, self.sce.nUEs, self.sce.nRBs), )
        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index].squeeze()
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[b_index, global_u_index, rb_index] / 10))

        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
        interference_m = interference_sum_m - signal_power_set + Noise
        log2input = 1 + signal_power_set / interference_m
        log2input[log2input <= np.zeros_like(log2input)] = 1
        unscale_rate_m = np.log2(log2input)
        # unscale_rate_m = np.log2(1 + 3)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)
        total_rate_m = self.sce.BW * unscale_rate_m / (10 ** 6)
        return total_rate, total_rate_m

    def MARLstep_deprecated(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:
        # todo action是基于obs(t)给出的 输入到step里后 输出的reward 是基于 obs(t+1)算的，这是否合理
        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """
        # reward, terminated, truncated, info = 1, False, False, {'actions': actions_user_association_RBG_assignment}
        # obs = self.observation_space.sample()
        # return obs, reward, terminated, truncated, info
        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        # .reshape(self.sce.nBSs, self.sce.nRBs)
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise

        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * signal_power
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power
        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise + 1e-7
        interference_m = np.clip(interference_m, a_min=1e-8, a_max=None)
        ratio = 1 + signal_power_set / interference_m
        ratio = np.clip(ratio, a_min=1e-8, a_max=None)
        unscale_rate_m = np.log2(ratio)
        # unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs
        # if not isinstance(actions_RBG_assignment_total, th.Tensor):
        #     actions_RBG_assignment_total = th.tensor(actions_RBG_assignment_total)
        # terminated, truncated, info = self.terminate_condtion(actions_RBG_assignment_total), False, {}
        # self.LastAction = actions_RBG_assignment_total
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info

    def get_init_obs(self, actions_user_association_RBG_assignment: np.ndarray):
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))
        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power
        obs = channal_power_set.reshape(1, -1)
        return obs

    def MARLreset(self, seed=None, options=None):

        # obs = np.ones(shape=(len(self.BSs), len(self.UEs)), dtype=self.dtype)
        # obs, _, _, _, _ = self.MARLstep(th.tensor(self.action_space.sample()))
        ua_act = []
        for i in self.user_association_observation_spaces:
            ua_act.append(i.sample())
        ua_act = np.concatenate(ua_act)

        rbg_act = []
        for i in self.rbg_assignment_observation_spaces:
            rbg_act.append(i.sample())
        rbg_act = np.concatenate(rbg_act)
        act = np.concatenate([ua_act, rbg_act])

        obs = self.get_init_obs(act)
        obs = np.squeeze(obs)
        import numpy as _np

        # 1) 用零填充初始的 channel 信息：shape = (BS_num, nUEs, nRBs)
        self.history_channel_information = _np.zeros(
            (self.BS_num, self.sce.nUEs, self.sce.nRBs),
            dtype=self.dtype
        )

        # 2) 将其展平成 1D 向量返回
        obs = self.history_channel_information.ravel()
        info = {}
        return obs, info


class MarlCustomEnv5(MarlCustomEnv4):
    # change actionspace and obs space
    # change MARLsetp logits
    # before this version, the marlstep use the next time slot obs to cal the current action's reward which is weired
    # now, we store the obs in the self.history_channel_information then use it to calculate the reward
    # the new function has the notation called MARLstep_withCurrentH
    # you can search the word : MARLstep_withCurrentH
    def __init__(self, sce, env_instance: MarlCustomEnv = None):
        if env_instance is None:
            super(MarlCustomEnv, self).__init__(sce)
        else:
            self.__setstate__(env_instance.__getstate__())
        # TODO check Encoder
        self.history_channel_information = None
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        # self.TxPowerVector = np.array([b.Transmit_Power_dBm() for b in self.BSs])
        # b, k, u = self.BS_num, sce.nRBs, sce.nUEs
        # self.MatrixE = np.zeros((u, k * u))
        # for every basestation, its user candidate are the set of user locating in its signal coverage zone.
        # thus, the action space of every agent is the box with dimension of number of users in its signal coverage zone.

        self.user_association_action_spaces = []
        self.rbg_assignment_action_spaces = []
        self.user_association_observation_spaces = []
        self.rbg_assignment_observation_spaces = []
        # self.user_association_observation_spaces = []
        # self.rbg_assignment_observation_spaces = []
        # totalActionDimension_ua=0
        # totalActionDimension_rbg=0
        totalActionDimension = 0

        for b in self.BSs:
            # for actor policy input parameter
            totalActionDimension += len(b.UE_set) + len(b.UE_set) * self.sce.nRBs
            # totalActionDimension_ua += len(b.UE_set)
            self.user_association_action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set),)))
            self.user_association_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (len(b.UE_set),), dtype=self.dtype))

            # totalActionDimension_rbg += len(b.UE_set) * self.sce.nRBs
            self.rbg_assignment_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) * (self.sce.nRBs + 1),)))
            self.rbg_assignment_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf,
                                   (len(b.UE_set) * self.sce.nRBs,),
                                   dtype=self.dtype)
            )

        # for replay buffer
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf,
                                                    shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
                                                    dtype=self.dtype)

        # self.action_space_ua = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension_ua,))
        # self.observation_space_ua = gym.spaces.box.Box(low=-np.inf, high=np.inf,
        #                                             shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
        #                                             dtype=self.dtype)
        #
        # self.action_space_rbg = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension_rbg,))
        # self.observation_space_rbg = gym.spaces.box.Box(low=-np.inf, high=np.inf,
        #                                             shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
        #                                             dtype=self.dtype)

        # if we change the dimension of observation_space,
        # the item list needed to be modified is as follows:
        # self.observation_space(s)
        # distribution_obs() distribution_actions()
        # self.predict4()
        # for critic input parameter
        self.share_critic_action_spaces = []
        self.share_critic_observation_spaces = []

        for b in self.BSs:
            self.share_critic_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) + len(b.UE_set) * self.sce.nRBs,)))
            self.share_critic_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (totalActionDimension // (self.sce.nRBs + 1) * self.sce.nRBs,),
                                   dtype=self.dtype))

    def step(self, actions):
        # return self.RLstep(actions)
        # return self.Pathloss_3GPP_RLStep(actions)
        # return self.step_with_cost(actions)
        return self.MARLstep_withCurrentH(actions)

    def MARLstep_withCurrentH(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:
        # MARLstep_withCurrentH
        # todo action是基于obs(t)给出的 输入到step里后 输出的reward 是基于 obs(t+1)算的，这是否合理
        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """

        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        # for not FACMAC5
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))
        # if self.use_action_projection:
        #     bs_wise_RBG_assignment_action = self.project_rbg_action(bs_wise_RBG_assignment_action)
        # only for facmac5
        bs_wise_RBG_assignment_action = []
        bs_wise_user_association_action = []
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_RBG_assignment[..., index:index + stride])
            bs_wise_RBG_assignment_action.append(
                actions_user_association_RBG_assignment[...,
                index + stride:index + stride * (self.sce.nRBs + 1)].reshape(-1, stride, self.sce.nRBs))
            index += stride * (self.sce.nRBs + 1)
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise

        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理
        assert self.history_channel_information is not None
        # if self.history_channel_information is not None:
        H_dB = self.history_channel_information.reshape((self.BS_num, self.sce.nUEs, self.sce.nRBs), )
        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index].squeeze()
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    _, channel_power_dBm \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    # s_b_u *
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[b_index, global_u_index, rb_index] / 10))
                    # print(f'sbku before:{signal_power_set[rb_index][global_u_index]}\nafter{signal_power_set[rb_index][global_u_index]*s_b_u}')
                    # H[b_index,global_u_index,rb_index] * b.Transmit_Power_dBm()
                    # 注意 H_dB 是 fading - pathloss
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm
        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs
        # if not isinstance(actions_RBG_assignment_total, th.Tensor):
        #     actions_RBG_assignment_total = th.tensor(actions_RBG_assignment_total)
        # terminated, truncated, info = self.terminate_condtion(actions_RBG_assignment_total), False, {}
        # self.LastAction = actions_RBG_assignment_total
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info


class TerminatedState(gym.Wrapper):
    def __init__(self, env, max_steps=None):
        super(TerminatedState, self).__init__(env)
        self.total_steps = 0
        self.max_steps = max_steps

    def step(self, action):

        observation, reward, done, truncated, info = self.env.step(action)
        self.total_steps += 1
        if self.max_steps is None or self.total_steps < self.max_steps:
            done = False
        else:
            self.total_steps = 0
            done = True

        return observation, reward, done, truncated, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        # self edit
        observation, reward, done, truncated, info = self.env.step(action)
        # original
        # observation, reward, terminated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
            # self edit
            truncated = True
        # self edit
        return observation, reward, done, truncated, info
        # orignal
        # return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def learning_rate_schedule(progress):
    initial_learning_rate = 2.5e-4
    final_learning_rate = 2.5e-5
    return initial_learning_rate * (1 - progress) + final_learning_rate * progress


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    import json

    sce = DotDic(json.loads(open('../Config/config_1.json', 'r').read()))
    opt = DotDic(json.loads(open('../Config/config_2.json', 'r').read()))  # Load the configuration file as arguments
    trial_opt = copy.deepcopy(opt)
    trial_sce = copy.deepcopy(sce)
    # env = MarlCustomEnv(trial_sce, trial_opt)

    env_user_association = load_env(
        "../Experiment_result/FACMAC/args1/date20240606time175309/model_saves/env.zip")
    env = MarlCustomEnv2(env_user_association.sce,
                         opt,
                         env_user_association
                         )
    actions = env.action_space.sample()
    env.MARLstep(actions)

    check_env(env)
# from stable_baselines3.common.envs import check_envs
