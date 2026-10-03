
#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# 自动添加项目根目录到sys.path，保证mat包可被正确导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mat.config import get_config
from mat.envs.rrm.RRM_env import RRMEnv
from mat.runner.shared.rrm_runner import RRMRunner
from mat.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from mat.envs.env_wrappers import ShareDummyVecEnv
import yaml


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "RRM":
                env = RRMEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # 不同环境不同种子
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:#单进程和多进程
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "RRM":
                env = RRMEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--config_env', type=str,
                        default='config_environment_setting_MAT.yaml',
                        help="path to yaml env config")

    parser.add_argument('--n_mbs', type=int, default=0, help="number of macro base stations")
    parser.add_argument('--n_pbs', type=int, default=5, help="number of pico base stations")
    parser.add_argument('--n_fbs', type=int, default=0, help="number of femto base stations")
    parser.add_argument('--n_ues', type=int, default=50, help="number of user equipments")
    parser.add_argument('--n_channels', type=int, default=5, help="number of channels")
    parser.add_argument('--r_mbs', type=float, default=500, help="radius of macro base station")
    parser.add_argument('--r_pbs', type=float, default=300, help="radius of pico base station")
    parser.add_argument('--r_fbs', type=float, default=100, help="radius of femto base station")
    parser.add_argument('--txpower_mbs_dbm', type=float, default=43, help="transmit power of macro base station in dBm")
    parser.add_argument('--txpower_pbs_dbm', type=float, default=36, help="transmit power of pico base station in dBm")
    parser.add_argument('--txpower_fbs_dbm', type=float, default=23, help="transmit power of femto base station in dBm")
    parser.add_argument('--bandwidth', type=float, default=180e3, help="channel bandwidth in Hz")
    parser.add_argument('--noise_power', type=float, default=-174, help="noise power in dBm/Hz")
    parser.add_argument('--frequency', type=float, default=2.5, help="carrier frequency in GHz")
    parser.add_argument('--x_max', type=float, default=1000, help="maximum x coordinate")
    parser.add_argument('--y_max', type=float, default=1000, help="maximum y coordinate")
    parser.add_argument('--print_config', action='store_true', default=False, help="whether to print environment configuration")
    parser.add_argument('--obs_dim', type=int, default=30, help="dimension of observation space")
    
    # 添加预训练模型相关参数
    parser.add_argument('--pretrained_model_path', type=str, default=None, help="path to pretrained model")
    parser.add_argument('--use_pretrain', action='store_true', default=False, help="whether to use pretrained model")

    # 显式设备选择（可覆盖全局 --cuda 逻辑）
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help='override device: cpu or cuda (default: auto by --cuda & availability)')
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.use_eval = True
    print(f"Forced use_eval to: {all_args.use_eval}")

    # 1) load env config（支持在 train 子目录运行时自动定位到 mat/scripts/ 下的配置文件）
    cfg_candidates = []
    # 用户传入的原始路径（可能是相对路径或绝对路径）
    cfg_candidates.append(all_args.config_env)
    # 相对当前工作目录
    cfg_candidates.append(os.path.join(os.getcwd(), all_args.config_env))
    # 相对脚本目录的上一级（mat/scripts/）
    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_candidates.append(os.path.join(scripts_dir, all_args.config_env))
    # 明确指向 mat/scripts 下（避免多级相对路径问题）
    cfg_candidates.append(os.path.join(project_root, 'mat', 'scripts', all_args.config_env))

    config_env_file = None
    for cand in cfg_candidates:
        if os.path.isfile(cand):
            config_env_file = os.path.abspath(cand)
            break
    if config_env_file is None:
        tried = "\n  - " + "\n  - ".join(cfg_candidates)
        raise FileNotFoundError(
            f"配置文件未找到: {all_args.config_env}\n尝试位置:{tried}\n"
            f"请传入 --config_env 的正确路径，或将文件放在 mat/scripts/ 下。"
        )

    with open(config_env_file, 'r') as f:
        env_cfg = yaml.safe_load(f)

    # 打印 RRM 环境配置
    if all_args.print_config:
        print("===== RRM 环境配置 =====")
        for key, val in env_cfg.items():
            print(f"{key}: {val}")
        print("========================")

    # 2) override all_args
    all_args.n_ues           = env_cfg['nUEs']
    all_args.n_rbs           = env_cfg['nRBs']
    all_args.n_mbs           = env_cfg['nMBS']
    all_args.n_pbs           = env_cfg['nPBS']
    all_args.n_fbs           = env_cfg['nFBS']
    all_args.r_mbs           = env_cfg['rMBS']
    all_args.r_pbs           = env_cfg['rPBS']
    all_args.r_fbs           = env_cfg['rFBS']
    all_args.txpower_mbs_dbm = env_cfg['txpowerMBSdBm']
    all_args.txpower_pbs_dbm = env_cfg['txpowerPBSdBm']
    all_args.txpower_fbs_dbm = env_cfg['txpowerFBSdBm']
    all_args.bandwidth       = env_cfg['BW']
    all_args.n_channel       = env_cfg['nChannel']
    all_args.noise_power     = env_cfg['N0']
    all_args.qos_thr         = env_cfg['QoS_thr']
    all_args.fc              = env_cfg['fc']
    all_args.x_max           = env_cfg['x_max']
    all_args.y_max           = env_cfg['y_max']
    all_args.nb              = env_cfg['Nb']
    all_args.nrb_max         = env_cfg['Nrb']

    # 3) 重算基站数量 & agents
    all_args.num_agents = (all_args.n_mbs
                           + all_args.n_pbs
                           + all_args.n_fbs)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # 设备选择逻辑：优先使用 --device 覆盖，其次按 --cuda 和可用性自动选择
    device = None
    if all_args.device is not None:
        if all_args.device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                print("[WARN] --device cuda 指定，但未检测到可用的 CUDA，回退到 CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    else:
        if all_args.cuda and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    # 日志与后续优化器/线程配置
    if device.type == 'cuda':
        name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'Unknown CUDA device'
        print(f"Using CUDA on {name}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True
        )
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                               str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    # 提取第一个子环境的 local_obs_dims
    if hasattr(envs, "envs"):
        local_obs_dims = envs.envs[0].max_local_obs_dim
    else:
        # 如果是 SubprocVecEnv，则：
        local_obs_dims = envs.venv.envs[0].max_local_obs_dim

    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    # num_agents = 3  # 3 base stations

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from mat.algorithms.mat.mat_trainer import MATTrainer as TrainerSingle
    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as PolicySingle
    from mat.utils.shared_buffer import SharedReplayBuffer

    all_args.num_agents = (all_args.n_mbs
                           + all_args.n_pbs
                           + all_args.n_fbs)

    # ← 在这里定义局部 num_agents 变量
    num_agents = all_args.num_agents

    # 策略网络
    if all_args.share_policy:
        policy = PolicySingle(all_args, 
                             envs.observation_space[0], 
                             envs.share_observation_space[0],
                             envs.action_space[0], 
                             num_agents,
                             local_obs_dims,
                             device)
    else:
        raise NotImplementedError

    # 加载预训练模型
    if all_args.use_pretrain and all_args.pretrained_model_path:
        print(f"Loading pretrained model from: {all_args.pretrained_model_path}")
        try:
            checkpoint = torch.load(all_args.pretrained_model_path, map_location=device)
            
            # Check if it's a MAT-bridged model (our converted model)
            if isinstance(checkpoint, dict) and 'transformer_state_dict' in checkpoint:
                # This is a MAT-compatible checkpoint (either from our bridge or from MAT training)
                policy.transformer.load_state_dict(checkpoint['transformer_state_dict'], strict=False)
                print("Successfully loaded MAT-compatible pretrained model!")
                
                # Show transfer information if available
                if 'transfer_info' in checkpoint:
                    transfer_info = checkpoint['transfer_info']
                    print(f"Transfer method: {transfer_info.get('transfer_method', 'unknown')}")
                    print(f"Original model path: {transfer_info.get('simple_model_path', 'unknown')}")
                    print(f"Model dimensions - obs: {transfer_info.get('obs_dim')}, share_obs: {transfer_info.get('share_obs_dim')}, action: {transfer_info.get('action_dim')}")
                
                print(f"Pretrained model epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"Pretrained model val loss: {checkpoint.get('val_loss', 'unknown')}")
                
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Check if the model architecture is compatible with MAT
                model_state = checkpoint['model_state_dict']
                mat_keys = set(policy.transformer.state_dict().keys())
                pretrained_keys = set(model_state.keys())
                
                # Check for key compatibility
                common_keys = mat_keys.intersection(pretrained_keys)
                if len(common_keys) < len(mat_keys) * 0.1:  # Less than 10% overlap
                    print("WARNING: Pretrained model architecture is incompatible with MAT architecture!")
                    print(f"MAT has {len(mat_keys)} keys, pretrained model has {len(pretrained_keys)} keys")
                    print(f"Only {len(common_keys)} keys match")
                    print("The pretrained model appears to be from a different architecture (e.g., supervised learning model)")
                    print("SUGGESTION: Use bridge_pretrained_to_mat.py to convert the model to MAT format")
                    print("Continuing with random initialization...")
                else:
                    # Load compatible weights
                    policy.transformer.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("Successfully loaded compatible pretrained model weights!")
                    print(f"Loaded {len(common_keys)} out of {len(mat_keys)} parameters")
                    print(f"Pretrained model epoch: {checkpoint.get('epoch', 'unknown')}")
                    print(f"Pretrained model val loss: {checkpoint.get('val_loss', 'unknown')}")
                    
            elif isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                # Load from RL checkpoint
                policy.transformer.load_state_dict(checkpoint['policy_state_dict'], strict=False)
                print("Successfully loaded pretrained model from RL checkpoint!")
                print(f"Pretrained model epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"Pretrained model val loss: {checkpoint.get('val_loss', 'unknown')}")
            else:
                # Assume it's just the state dict directly
                policy.transformer.load_state_dict(checkpoint, strict=False)
                print("Successfully loaded pretrained model!")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            print("Continuing with random initialization...")
    elif all_args.use_pretrain:
        print("Warning: --use_pretrain is True but --pretrained_model_path is not provided!")
        print("Continuing with random initialization...")

    buffer = SharedReplayBuffer(
        all_args,
        num_agents,
        envs.observation_space[0],
        envs.share_observation_space[0]
            if all_args.use_centralized_V else envs.observation_space[0],
        envs.action_space[0],
        all_args.env_name
    )

    trainer = TrainerSingle(
        all_args,
        policy,
        num_agents,
        device=device
    )

    from tensorboardX import SummaryWriter
    
    if all_args.use_wandb:
        writter = SummaryWriter(str(run_dir))
    else:
        writter = SummaryWriter(str(run_dir))
        
    config["buffer"] = buffer
    config["trainer"] = trainer
    config["writter"] = writter
    config["log_dir"] = run_dir
    
    runner = RRMRunner(config)
    
    runner.run()
    
    envs.close()
    if all_args.use_eval:
        eval_envs.close()
        
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir / 'summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])