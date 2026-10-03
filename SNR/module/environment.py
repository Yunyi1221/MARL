# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from random import random, uniform, choice, randrange
import numpy as np
from numpy import pi, sqrt, log10
import torch
import gymnasium as gym
import random
'''
@Project ：UARL
@File    ：environment.py
@Author  ：Jiansheng LI
@Date    ：2023/11/27 14:12
'''

class BS:  # Define the base station (Agent)
    def __init__(self, sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm):
        self.sce = sce
        self.id = BS_index
        self.BStype = BS_type
        self.BS_Loc = BS_Loc
        self.BS_Radius = BS_Radius
        self.UE_set = []  # 与其关联的用户结合

        self.Tx_Power_dBm = Tx_Power_dBm
        # self.Tx_Power = 10 ** (Tx_Power_dBm / 10)
        # def __getstate__(self):

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def reset(self):  # Reset the channel status
        self.Ch_State = np.zeros(self.sce.nChannel)

    def Get_Location(self):
        return self.BS_Loc

    def Transmit_Power(self):  # Calculate the transmit power of a BS
        # if self.BStype == "MBS":
        #     Tx_Power_dBm = 40
        # elif self.BStype == "PBS":
        #     Tx_Power_dBm = 30
        # elif self.BStype == "FBS":
        #     Tx_Power_dBm = 20
        return 10 ** (self.Tx_Power_dBm / 10)  # Transmit power in dBm, no consideration of power allocation now

    def Transmit_Power_dBm(self):  # Calculate the transmit power of a BS
        # if self.BStype == "MBS":
        #     Tx_Power_dBm = 40
        # elif self.BStype == "PBS":
        #     Tx_Power_dBm = 30
        # elif self.BStype == "FBS":
        #     Tx_Power_dBm = 20
        return self.Tx_Power_dBm  # Transmit power in dBm, no consideration of power allocation now

    def Select_Action(self, state, scenario, eps_threshold):  # Select action for a user based on the network state
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels
        sample = random()
        if sample < eps_threshold:  # epsilon-greeedy policy
            with torch.no_grad():
                Q_value = self.model_policy(state)  # Get the Q_value from DNN
                action = Q_value.max(0)[1].view(1, 1)
        else:
            action = torch.tensor([[randrange(L * K)]], dtype=torch.long)
        return action
class UE:  # Define the agent (UE)
    def __init__(self, x, y, index):  # Initialize the agent (UE)
        self.id = index
        self.location = [x, y]
        self.original_location = [x, y]
        self.BS_set = [] #其关联的基站
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.original_location = state['location']

    # def Set_Location(self, BS_location, BS_type : str):  # Initialize the location of the agent
    #     #
    #     #     LocM = BS_location
    #     #     Loc_agent = np.zeros(2)  # 预设agent的位置
    #     #     # Return a random element from a list: 随机选一个MBS
    #     #
    #     #     r = self.sce['r'+BS_type] * random()  # MBS的半径 500米
    #     #     theta = uniform(-pi, pi)  # theta
    #     #     Loc_agent[0] = LocM[0] + r * np.cos(theta)  # 随机选取位置
    #     #     Loc_agent[1] = LocM[1] + r * np.sin(theta)  #
    #     #     return Loc_agent

    def Get_Location(self):
        return self.location

    # def sample(self):
    #     '''Only sample from the remaining valid spaces
    #     '''
    #     if len(self.valid_spaces) == 0:
    #         print("Space is empty")
    #         return None
    #     np_random, _ = seeding.np_random()
    #     randint = np_random.randint(len(self.valid_spaces))
    #     return self.valid_spaces[randint]


# gym.Env
class Environment(gym.Env):
    def __init__(self, sce: dict):  # Initialize the scenario we simulate
        self.seed = 777
        self.sce = sce
        self.BS_num = self.sce.nMBS + self.sce.nPBS + self.sce.nFBS
        self.nBSs = self.BS_num
        self.nRBs = self.sce.nRBs
        self.nUEs= self.sce.nUEs
        self.BSs = self.BS_Init()
        self.UEs = self.UE_Init()
        # self.user_candidate_assignment()
        self.distance_matrix = self.get_distance_matrix()
        self.burst_prob = sce.get("burst_prob", None)  # 用户数据请求的概率
        self.isBurstScenario = True
        self.eval_mode = False
        if self.isBurstScenario and self.burst_prob is not None:
            # assert self.burst_prob is not None
            print("=" * 10, 'Note: this is the burst scenario', f"self.burst_prob: {self.burst_prob}", "=" * 10)
        else:
            self.isBurstScenario = False
        self.init_ue_serving_BSs()
        self.PH = self.get_new_CSI() # 获得一组功率*信道的实例
        if sce.prt:
            self.showmap()

    def init_ue_serving_BSs(self):
        """
        根据距离确定的BS-UE pair, 即用户关联(为BS添加其关联的UE集合信息, 为UE加入其关联的基站id信息)
        :return:
        """
        # 初始化所有BS的UE_set为空列表
        for b in self.BSs:
            b.UE_set = []

        # 初始化所有UE的BS_set为空列表
        for u in self.UEs:
            u.BS_set = []

        # 对于每个UE，找出距离最近的BS
        for u_index, u in enumerate(self.UEs):
            min_distance = float('inf')
            serving_bs = None

            # 遍历所有BS，找到最近的BS
            for b_index, b in enumerate(self.BSs):
                dist = self.distance_matrix[b_index][u_index]
                if dist < min_distance and dist <= b.BS_Radius:
                    min_distance = dist
                    serving_bs = b

            # 如果找到合适的BS，则建立关联关系
            if serving_bs:
                # 将UE添加到BS的UE_set中
                serving_bs.UE_set.append(u.id)

                # 将BS添加到UE的BS_set中
                u.BS_set.append(serving_bs.id)

        # 确保所有用户都有关联的基站（如果没有，则强制关联最近的基站）
        for u in self.UEs:
            if not u.BS_set:
                min_distance = float('inf')
                serving_bs = None

                for b_index, b in enumerate(self.BSs):
                    dist = self.distance_matrix[b_index][self.UEs.index(u)]
                    if dist < min_distance:
                        min_distance = dist
                        serving_bs = b

                if serving_bs:
                    serving_bs.UE_set.append(u.id)
                    u.BS_set.append(serving_bs.id)

    def get_distance_matrix(self):
        distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
        print('distance matrix is calculated!')
        return distance_matrix

    def get_n0(self):
        # return the environmental noise # note: not dB
        return 10 ** (float(self.sce.N0) / 10) * float(self.sce.BW)
    def get_n0_dB(self):
        # return the environmental noise # note: not dB
        return self.sce.N0 * self.sce.BW

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.distance_matrix = self.get_distance_matrix()

    def showmap(self, path: str = None):
        """
        Visualize the base and user location
        :param path:  path to save the figure
        :return:
        """
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，避免GUI窗口
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        # if path is None:
        #     path = "../my_env_plot.png"
        for b in self.BSs:
            print(f'BS id: {b.id}, UE_Set: {b.UE_set}\n')
        base_positions = []
        user_positions = []
        user_original_postions = []
        for b in self.BSs:
            base_positions.append(b.BS_Loc)
        for u in self.UEs:
            try:
                user_original_postions.append(u.original_location)
                user_positions.append(u.location)
            except Exception as e:
                user_original_postions.append(u.location)
                user_positions.append(u.location)
        # 设定基站的位置
        # base_positions = np.array([[20, 20], [40, 40], [60, 60], [80, 80], [100, 100]])
        base_positions = np.array(base_positions)

        # 基站的信号覆盖半径
        # def generate_base_positions(n):
        #     if n < 1 or (np.sqrt(8 * n - 7) - 1) / 3 % 1 != 0:
        #         raise ValueError("基站的数量必须是一个完全六边形数，如1、7、19、37")
        #     # 计算m的值
        #     m = int((np.sqrt(8 * n - 7) + 1) // 3)
        #     # 初始化基站位置列表
        #     base_positions = []
        #     for i in range(-m + 1, m):
        #         for j in range(max(-m + 1, -i - m + 1), min(m, -i + m)):
        #             x = i + j / 2
        #             y = j * np.sqrt(10) / 2
        #             base_positions.append([x, y])
        #     return np.array(base_positions)

        # base_positions=generate_base_positions(7)
        # radius = 0.6

        # 每个基站的用户数量
        # num_users_per_base = 10

        # 定义矩形区域的边界
        # x_min, x_max = 0, 1000
        # y_min, y_max = 0, 1000

        # 定义要生成的点的数量
        # num_points = 50

        # 生成均匀分布的随机 x 和 y 坐标
        # x = np.random.uniform(x_min, x_max, num_points)
        # y = np.random.uniform(y_min, y_max, num_points)

        # 用户的位置
        # user_positions = []
        #
        # for base_position in base_positions:
        #     for _ in range(num_users_per_base):
        #         # 在基站的信号覆盖范围内随机选择一个角度和半径
        #         angle = 2 * np.pi * np.random.rand()
        #         r =  100* np.sqrt(np.random.rand())
        #         # 计算用户的位置
        #         user_position = base_position + np.array([r * np.cos(angle), r * np.sin(angle)])
        #         user_positions.append(user_position)
        #

        user_positions = np.array(user_positions)
        user_original_postions = np.array(user_original_postions)
        # 画出基站和用户的位置
        plt.scatter(user_positions[:, 0], user_positions[:, 1], label='user', color='blue')
        # plt.scatter(x, y, label='user')

        plt.scatter(base_positions[:, 0], base_positions[:, 1], label='basestation', color='red')
        # 获取当前的Axes对象
        ax = plt.gca()

        # 在每个点上绘制圆圈
        for idx, (xi, yi), in enumerate(zip(base_positions[:, 0], base_positions[:, 1])):
            circle = Circle((xi, yi), self.BSs[idx].BS_Radius, color='darkred', fill=False, linewidth=1.5, alpha=0.6)
            ax.add_patch(circle)
        # todo radius = 50 ?
        # for idx, (xi, yi), in enumerate(zip(user_original_postions[:, 0], user_original_postions[:, 1])):
        #     circle = Circle((xi, yi), 50, color='darkred', fill=False, linewidth=1, alpha=0.6)
        #     ax.add_patch(circle)
        # 设置坐标轴比例相同，以保持圆形
        ax.set_aspect('equal', adjustable='datalim')
        plt.legend()
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            print('Map figure saved to ' + path)
        plt.close()  # 关闭图形，避免GUI阻塞

    def user_candidate_assignment(self) -> None:
        # add UE in every BS's UE candidate set, where UE added in the
        # set is in the signal coverage zone of the BS.
        b: BS
        u: UE
        for b in self.BSs:
            for u in self.UEs:
                if sqrt((u.location[0] - b.BS_Loc[0]) ** 2 + (u.location[1] - b.BS_Loc[1]) ** 2) <= b.BS_Radius:
                    b.UE_set.append(u.id)
        return

    def BS_Init(self) -> list:  # Initialize all the base stations
        BaseStations = []  # The vector of base stations
        Loc_MBS, Loc_PBS, Loc_FBS = self.BS_Location()

        for i in range(self.sce.nMBS):  # Initialize the MBSs
            BS_index = i
            BS_type = "MBS"
            BS_Loc = Loc_MBS[i]
            BS_Radius = self.sce.rMBS
            Tx_Power_dBm = self.sce.txpowerMBSdBm
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm))

        for i in range(self.sce.nPBS):
            BS_index = self.sce.nMBS + i
            BS_type = "PBS"
            BS_Loc = Loc_PBS[i]
            BS_Radius = self.sce.rPBS
            Tx_Power_dBm = self.sce.txpowerPBSdBm
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm))

        for i in range(self.sce.nFBS):
            BS_index = self.sce.nMBS + self.sce.nPBS + i
            BS_type = "FBS"
            BS_Loc = Loc_FBS[i]
            BS_Radius = self.sce.rFBS
            Tx_Power_dBm = self.sce.txpowerFBSdBm
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm))

        return BaseStations

    def BS_Location(self):
        x_max = self.sce.x_max
        y_max = self.sce.y_max

        # 宏基站：六边形环绕布局
        Loc_MBS = np.zeros((self.sce.nMBS, 2))
        if self.sce.nMBS == 7:  # 标准7基站拓扑
            d = 800  # 基站间距
            center = [x_max / 2, y_max / 2]
            angles = [0, 60, 120, 180, 240, 300]  # 60°间隔

            # 中心基站
            Loc_MBS[0] = center

            # 周围6个基站
            for i in range(6):
                angle_rad = np.deg2rad(angles[i])
                Loc_MBS[i + 1] = [
                    center[0] + d * np.cos(angle_rad),
                    center[1] + d * np.sin(angle_rad)
                ]
        else:  # 其他数量的通用布局
            for i in range(self.sce.nMBS):
                Loc_MBS[i, 0] = x_max * i / max(1, self.sce.nMBS - 1)
                Loc_MBS[i, 1] = y_max / 2

        # # PBS/FBS：保持随机布局
        Loc_PBS = np.zeros((self.sce.nPBS, 2))
        Loc_FBS = np.zeros((self.sce.nFBS, 2))
        #
        # for i in range(self.sce.nPBS):
        #     Loc_PBS[i, 0] = uniform(0, x_max)
        #     Loc_PBS[i, 1] = uniform(0, y_max)
        #
        # for i in range(self.sce.nFBS):
        #     Loc_FBS[i, 0] = uniform(0, x_max)
        #     Loc_FBS[i, 1] = uniform(0, y_max)

        return Loc_MBS, Loc_PBS, Loc_FBS

    # 用户均匀分布初始化
    def UE_Init(self) -> list:
        # ------------- 完全均匀地散布在区域内 --------------
        # UEs = []
        # x_max = self.sce.x_max
        # y_max = self.sce.y_max
        #
        # # 均匀分布在矩形区域
        # grid_size = int(np.ceil(np.sqrt(self.sce.nUEs)))
        # for i in range(self.sce.nUEs):
        #     # 网格化均匀分布
        #     row = i // grid_size
        #     col = i % grid_size
        #
        #     x = (col + 0.5) * (x_max / grid_size)
        #     y = (row + 0.5) * (y_max / grid_size)
        #     UEs.append(UE(x, y, index=i + 1))
        # ------------- 完全均匀地散布在区域内 --------------

        # ------------- 随机均匀地散布在区域内 --------------
        """
        这种拉丁超立方采样方法：
        将区域划分为 n × n 个子网格
        在每个子网格内随机采样一个点
        打乱所有点的顺序
        为每个用户分配一个点
        :return:
        """
        ue_index = 0
        UEs = []
        x_max = self.sce.x_max
        y_max = self.sce.y_max

        # 使用拉丁超立方采样确保高度均匀的随机分布
        grid_size = min(int(np.ceil(np.sqrt(self.sce.nUEs))), 100)  # 防止用户太多时分区过多

        # 创建采样点
        locations = []
        for i in range(self.sce.nUEs):
            # 计算子网格位置
            col = i % grid_size
            row = i // grid_size

            # 在每个子网格内随机采样一个点
            x_min = col * (x_max / grid_size)
            x_max_sub = (col + 1) * (x_max / grid_size)
            y_min = row * (y_max / grid_size)
            y_max_sub = (row + 1) * (y_max / grid_size)

            x = uniform(x_min, x_max_sub)
            y = uniform(y_min, y_max_sub)
            locations.append((x, y))

        # 打乱位置顺序
        random.shuffle(locations)

        # 创建用户
        for i in range(self.sce.nUEs):
            x, y = locations[i]
            UEs.append(UE(x, y, index=ue_index))
            ue_index += 1

        return UEs
    def BS_Location_202412(self):
        Loc_MBS = np.zeros((self.sce.nMBS, 2))  # Initialize the locations of BSs
        Loc_PBS = np.zeros((self.sce.nPBS, 2))
        Loc_FBS = np.zeros((self.sce.nFBS, 2))
        x_max = self.sce.x_max
        y_max = self.sce.y_max
        bsloclist = self.sce.bsloclist
        if bsloclist is not None:
            assert (self.sce.nMBS + self.sce.nPBS + self.sce.nFBS) == len(bsloclist)
            for i in range(self.sce.nPBS):
                Loc_PBS[i, 0] = bsloclist[i][0]
                Loc_PBS[i, 1] = bsloclist[i][1]

            return Loc_MBS, Loc_PBS, Loc_FBS

        for i in range(self.sce.nMBS):
            # 任意两个MBS之间的间距为x_max
            MBS_deviation = x_max * 0.9  # default 900
            # x_max, default 1000
            Loc_MBS[i, 0] = x_max / 2 + x_max * 0.9 * i  # x-coordinate
            Loc_MBS[i, 1] = y_max / 2  # y-coordinate

        for i in range(self.sce.nPBS):
            # 每个MBS 分配 perMBS 个 PBS, 那么必须满足 self.sce.nPBS > perMBS
            perMBS = 4
            Loc_PBS[i, 0] = Loc_MBS[int(i / perMBS), 0] + x_max / 4 * np.cos(pi / 2 * (i % perMBS))
            Loc_PBS[i, 1] = Loc_MBS[int(i / perMBS), 1] + y_max / 4 * np.sin(pi / 2 * (i % perMBS))

        for i in range(self.sce.nFBS):
            LocM = choice(Loc_MBS)
            r = self.sce.rMBS * random()
            theta = uniform(-pi, pi)
            Loc_FBS[i, 0] = LocM[0] + r * np.cos(theta)
            Loc_FBS[i, 1] = LocM[1] + r * np.sin(theta)

        return Loc_MBS, Loc_PBS, Loc_FBS

    def path_loss_UMa_deprecated(self, d_2d, isLOS=True):
        # reference from 3GPPTR38.901ChannelModelChapter_V6_200820.pdf
        # For dense urban areas with high-rise buildings, h_E can be set between 15-30 m
        # For suburban areas with smaller houses, lower h_E of 5-15 m is appropriate
        h_e = 15
        h_ut = 1.5
        h_bs = 25
        fc = self.sce.fc  # Ghz
        c = 3 * 10 ** 8
        d_prime_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc * 10 ** 9 / c  # Urban Macrocell (UMa)
        d_3d = np.sqrt(d_2d ** 2 + (h_bs - h_ut) ** 2)  # Convert to m

        if d_2d <= d_prime_bp:
            r1 = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(fc)
        else:
            r1 = 28.0 + 40 * np.log10(d_3d) + 20 * np.log10(fc) - 9 * np.log10(
                d_prime_bp ** 2 + (h_bs - h_ut) ** 2)
        if not isLOS:
            return max(r1, 13.54 + 39.08 * np.log10(d_3d) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5))
        else:
            return r1

    def prob_LOS(self, d):
        """
        LOS Probability
        :param d:
        :return:
        """
        if d <= 18:
            return 1
        elif d <= 1000:
            return 18 / d + np.exp(-d / 63) * (1 - 18 / d)
        else:
            return 0

    def cal_Receive_Power(self, bs, d):
        # channel model defined here
        p_LOS = self.prob_LOS(d)

        # if np.random.rand() < p_LOS:
        #     loss_dB_LOS = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # else:
        #     loss_dB_NLOS = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)
        # loss_dB = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)

        assert 0 <= p_LOS <= 1
        # loss_dB_LOS = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB_NLOS = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)

        loss_dB_LOS = 128.1 + 37.6 * np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB_NLOS = 128.1 + 37.6 * np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=False)

        # loss_dB_LOS = 5*np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB_NLOS = 5*np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=False)

        # loss_dB = p_LOS * loss_dB_LOS + (1 - p_LOS) * loss_dB_NLOS
        # loss_dB = loss_dB/2
        # loss_dB = 128.1 + 37.6 * np.log10(d / 1000) + 9
        loss_dB = loss_dB_LOS
        Tx_Power_dBm = bs.Transmit_Power_dBm()

        # 旧信道模型
        # if bs.BStype == "MBS" or bs.BStype == "PBS":
        #     loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
        # elif bs.BStype == "FBS":
        #     loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()

        H_power_dB = loss_dB
        # 不考虑基站服务半径
        Rx_power_dBm = Tx_Power_dBm - loss_dB  # Received power in dBm
        Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW

        # 考虑服务半径
        # if d <= bs.BS_Radius:
        #     Rx_power_dBm = Tx_Power_dBm - loss_dB  # Received power in dBm
        #     Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
        # else:
        #     Rx_power = 0.0
        return Rx_power, H_power_dB

    def get_shadow_fading_dB(self, isLOS):
        """
        :param isLOS: True or False depending on whether the shadow fading is line of sight or not
        :param d_c: correlation factor for shadow fading between users based on the distance.
        :return:
        """
        std_dev = 4 if isLOS else 6
        z_fading_raw = sqrt(1 / 2) * (np.random.normal(loc=0.0, scale=std_dev) + 1j * np.random.normal(loc=0.0, scale=std_dev))
        z_fading = abs(z_fading_raw)
        z_fading_dB = 10 * log10(z_fading ** 2)

        # 不考虑LOS时
        # temp_h = sqrt(1 / 2) * (np.random.normal(loc=0.0, scale=scale) + 1j * np.random.normal(loc=0.0, scale=scale))
        #
        # h = abs(temp_h)  # 复数绝对值即范数
        # h_dB = 10 * log10(h ** 2)

        return z_fading_dB

    def get_new_CSI(self):
        """
        :return:
        """
        print('Generating new CSI....')
        channal_power_set = np.zeros((self.nBSs, self.nUEs, self.nRBs))
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUEs):
                for rb_index in range(self.nRBs):
                    signal_power, channel_power \
                        = self.cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[b_index][global_u_index][rb_index] = signal_power
        return channal_power_set

# if __name__ == '__main__':

# sce, opt = get_config()
# env = Environment(sce, prt=1)
# 可视化环境的基站和用户位置
# base_positions=[]
# user_positions=[]
# for b in env.BSs:
#     base_positions.append(b.BS_Loc)
# for u in env.UEs:
#     user_positions.append(u.location)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设定基站的位置
# # base_positions = np.array([[20, 20], [40, 40], [60, 60], [80, 80], [100, 100]])
# base_positions=np.array(base_positions)
# # 基站的信号覆盖半径
#
#
#
# def generate_base_positions(n):
#     if n < 1 or (np.sqrt(8 * n - 7) - 1) / 3 % 1 != 0:
#         raise ValueError("基站的数量必须是一个完全六边形数，如1、7、19、37")
#
#     # 计算m的值
#     m = int((np.sqrt(8 * n - 7) + 1) // 3)
#
#     # 初始化基站位置列表
#     base_positions = []
#
#     for i in range(-m + 1, m):
#         for j in range(max(-m + 1, -i - m + 1), min(m, -i + m)):
#             x = i + j / 2
#             y = j * np.sqrt(10) / 2
#             base_positions.append([x, y])
#
#     return np.array(base_positions)
#
#
# # base_positions=generate_base_positions(7)
# # radius = 0.6
#
# # 每个基站的用户数量
# num_users_per_base = 10
#
# # 定义矩形区域的边界
# x_min, x_max = 0, 1000
# y_min, y_max = 0, 1000
#
# # 定义要生成的点的数量
# num_points = 50
#
# # 生成均匀分布的随机 x 和 y 坐标
# x = np.random.uniform(x_min, x_max, num_points)
# y = np.random.uniform(y_min, y_max, num_points)
#
# # 用户的位置
# # user_positions = []
# #
# # for base_position in base_positions:
# #     for _ in range(num_users_per_base):
# #         # 在基站的信号覆盖范围内随机选择一个角度和半径
# #         angle = 2 * np.pi * np.random.rand()
# #         r =  100* np.sqrt(np.random.rand())
# #         # 计算用户的位置
# #         user_position = base_position + np.array([r * np.cos(angle), r * np.sin(angle)])
# #         user_positions.append(user_position)
# #
# user_positions = np.array(user_positions)
#
# # 画出基站和用户的位置
# plt.scatter(user_positions[:, 0], user_positions[:, 1], label='user')
# # plt.scatter(x, y, label='user')
#
# plt.scatter(base_positions[:, 0], base_positions[:, 1], label='base', color='r')
# plt.legend()
# plt.show()
