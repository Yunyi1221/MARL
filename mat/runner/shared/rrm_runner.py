import time
import wandb
import numpy as np
import torch

# 将tensor转换为numpy数组
def _t2n(x):
    return x.detach().cpu().numpy()

class RRMRunner:
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.step_count = 0
        
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V # 是否使用集中Critic
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state # 是否仅使用局部观测
        self.num_env_steps = self.all_args.num_env_steps # 总步数
        self.episode_length = self.all_args.episode_length # 每个episode的步数
        self.n_rollout_threads = self.all_args.n_rollout_threads # 并行的环境数
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size # transformer的隐藏层大小
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N # RNN的隐藏层大小

        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        self.model_dir = self.all_args.model_dir

        if 'use_valuenorm' in self.all_args: # 对价值函数norm
            self.use_valuenorm = self.all_args.use_valuenorm
        else:
            self.use_valuenorm = False
        
        self.use_proper_time_limits = self.all_args.use_proper_time_limits
        self.use_max_grad_norm = self.all_args.use_max_grad_norm
        self.max_grad_norm = self.all_args.max_grad_norm
        self.clip_param = self.all_args.clip_param # ppo的clip
        self.ppo_epoch = self.all_args.ppo_epoch
        self.num_mini_batch = self.all_args.num_mini_batch
        self.data_chunk_length = self.all_args.data_chunk_length
        self.value_loss_coef = self.all_args.value_loss_coef
        self.entropy_coef = self.all_args.entropy_coef
        self.use_value_active_masks = self.all_args.use_value_active_masks
        self.use_policy_active_masks = self.all_args.use_policy_active_masks
        self.huber_delta = self.all_args.huber_delta
        
        self.trainer = config['trainer']
        
        self.buffer = config['buffer']
        
        self.writter = config['writter']
        self.log_dir = config['log_dir']

        print(f"Runner initialized with:")
        print(f"  use_eval: {self.use_eval}")
        print(f"  eval_interval: {self.eval_interval}")
        print(f"  eval_envs is None: {self.eval_envs is None}")
        print(f"  eval_episodes: {getattr(self.all_args, 'eval_episodes', 'Not set')}")
        
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes) # 学习率调整，线性

            for step in range(self.episode_length):
                # 根据策略生成动作和value
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                
                # 执行动作，获取下一个状态和奖励
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                # 数据放入buffer
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                
                self.insert(data)

                # 更新奖励，当前step的奖励加入到当前episode的奖励中
                for i, reward in enumerate(rewards):
                    train_episode_rewards[i] += reward[0]
                    if dones[i].all(): # 如果当前episode完成，重置
                        done_episodes_rewards.append(train_episode_rewards[i])
                        train_episode_rewards[i] = 0

            # 计算优势函数和reward然后更新
            self.compute()
            train_infos = self.train()
            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
             
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("Average episode rewards is {}.".format(aver_episode_rewards))
                    if self.use_wandb:
                        wandb.log({"average_episode_rewards": aver_episode_rewards}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("average_episode_rewards", {"average_episode_rewards": aver_episode_rewards}, total_num_steps)
                    done_episodes_rewards = []

                self.log_train(train_infos, total_num_steps)

            # 定期评估
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        '''
        初始化环境和buffer，如果不使用集中Critic，则共享观测和局部观测相同
        '''
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs[0].copy()
        if available_actions is not None and self.buffer.available_actions is not None:
            self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad() # 不计算梯度，因为不是训练
    def collect(self, step):
        self.trainer.prep_rollout() # 将Critic设为eval模式
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]) if self.buffer.available_actions is not None else None)
        # concatenate是将所有环境的观测拼接在一起,一次性计算

        # 先将tensor转换为numpy数组，然后再分割成每个环境的观测
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic # 当前状态的value和动作，选择动作的log概率，RNN的状态

    def insert(self, data):
        '''
        数据处理后放入buffer
        '''
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=(1, 2))  # Dones 为 [n_rollout_threads, num_agents, 1]

        # 修复：使用正确的索引方式重置RNN状态
        if np.any(dones_env):
            for i in range(self.n_rollout_threads):
                if dones_env[i]:
                    # 只重置已完成环境的RNN状态
                    rnn_states[i] = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    rnn_states_critic[i] = np.zeros((self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        # 创建episode的mask,1表示没有结束，0表示结束
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        # 修复：对于已完成的环境，将其所有智能体的mask设为0
        for i in range(self.n_rollout_threads):
            if dones_env[i]:
                masks[i, :, :] = 0

        # 对dones取反，表示智能体还在活动中
        active_masks = (~dones).astype(np.float32)
        
        # 处理异常终止，全1为正常
        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs, obs, rnn_states, rnn_states_critic,
            actions, action_log_probs, values, rewards, masks, 
            bad_masks, active_masks, available_actions
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []

        # 重置环境
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            # 将动作mask拼起来
            if isinstance(eval_available_actions, np.ndarray) and eval_available_actions.ndim > 1:
                avail_arg = np.concatenate(eval_available_actions)
            else:
                avail_arg = None

            # 获取动作，deterministic=True
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(
                    np.concatenate(eval_share_obs),
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    avail_arg,
                    deterministic=True
                )
            
            # 拆分到每个环境
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            #执行动作，获取新的观测和奖励，然后累积奖励
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=(1, 2))
            
            # 回合结束就重置RNN状态
            if np.any(eval_dones_env):
                # Replace boolean indexing with explicit loop
                for i in range(self.n_eval_rollout_threads):
                    if eval_dones_env[i]:
                        eval_rnn_states[i] = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            
            # 更新mask
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            for i in range(self.n_eval_rollout_threads):
                if eval_dones_env[i]:
                    eval_masks[i, :, :] = 0

            # 记录episode奖励
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0)[eval_i])
                    one_episode_rewards = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards,
                    'eval_max_episode_rewards': [np.max(eval_episode_rewards)]
                }
                
                self.log_env(eval_env_infos, total_num_steps)
                print("Evaluation: average episode rewards: {}.".format(np.mean(eval_episode_rewards)))
                break

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def compute(self):
        self.trainer.prep_rollout()

        # 准备下一步状态，取buffer中最后一个step的数据，拼在一起
        cent_obs = np.concatenate(self.buffer.share_obs[-1])
        obs = np.concatenate(self.buffer.obs[-1])
        rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[-1])
        masks = np.concatenate(self.buffer.masks[-1])
        # 如果有动作mask，也拼在一起
        if self.buffer.available_actions is not None:
            avail = np.concatenate(self.buffer.available_actions[-1])
            # 调用Critic前向，计算下一个状态的值
            next_values = self.trainer.policy.get_values(cent_obs, obs, rnn_states_critic, masks, avail)
        else:
            next_values = self.trainer.policy.get_values(cent_obs, obs, rnn_states_critic, masks)

        # 把tensor转换为numpy数组，然后分割成每个环境的观测，然后调用函数计算优势
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        self.trainer.policy.save(self.log_dir, episode)