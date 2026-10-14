# MARL: Multi-Agent Transformer for Wireless Resource Allocation

This repository applies the **Multi-Agent Transformer (MAT)** to joint
**user association and resource-block (RB) allocation** in a multi-cell
wireless network. Each base station is an agent. MAT treats the
base stations as a sequence and produces their actions one after another, so
each base station's decision is conditioned on the ones before it.

The MAT implementation is adapted from the official PKU code
([PKU-MARL/Multi-Agent-Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer),
Wen et al., *Multi-Agent Reinforcement Learning is a Sequence Modeling
Problem*, NeurIPS 2022). On top of it the repo adds a custom radio resource
management environment (`RRM`), a constraint projection step, an optional
local-search refinement, and a heuristic MAX-SNR baseline.

> **Status.** The wireless RRM work is no longer under active development.
> Follow-up work that reuses MAT's ideas (autoregressive agents, learnable
> agent order) on operations-research inventory problems lives in
> [MARL-in-Multy-Inventory-Problem](https://github.com/Yunyi1221/MARL-in-Multy-Inventory-Problem).

## Problem

Let $\mathcal{B}$ be the base stations, $\mathcal{U}$ the users (UEs) and
$\mathcal{K}$ the resource blocks. The decision variables are

- $z_{b,u} \in \{0,1\}$: user $u$ is associated with base station $b$;
- $a_{b,k,u} \in \{0,1\}$: base station $b$ serves user $u$ on RB $k$.

The objective is the total spectral efficiency

$$
\max_{z,a}\; \sum_{u \in \mathcal{U}} \sum_{b \in \mathcal{B}} \sum_{k \in \mathcal{K}} \log\left(1+\gamma_{b,k,u}\right),
\qquad
\gamma_{b,k,u} = \frac{z_{b,u}\, a_{b,k,u}\, P_{b,k,u} H_{b,k,u}}
{\sum_{b',\,u' \neq u} z_{b',u'}\, a_{b',k,u'}\, P_{b',k,u'} H_{b',k,u'} + n_0}
$$

subject to

| Constraint | Meaning |
|---|---|
| $\sum_b z_{b,u} \le 1$ | each user connects to at most one base station |
| $\sum_u z_{b,u} \le N_b$ | each base station serves at most $N_b$ users |
| $\sum_k a_{b,k,u} \le N_{rb}$ | each user gets at most $N_{rb}$ RBs |

Users on the same RB in different cells interfere with each other, which is why
the base stations have to coordinate. The full formulation and the
heuristic algorithm are written up (in Chinese) in [`SNR/MAX-SNR.md`](SNR/MAX-SNR.md).

## MARL formulation

| Element | Definition |
|---|---|
| Agent | one base station |
| Local observation | channel gains between this base station and the users in its coverage, shape `(users, nRBs)`, zero-padded to a common size |
| Shared observation | channel gains of all base stations, shape `(n_agents, nUEs, nRBs)` |
| Action | continuous vector in `[0, 1]`: one association score per covered user, followed by one score per (user, RB) pair; padded to the largest agent |
| Reward | total rate $\sum \log(1+\text{SINR})$, shared by all agents |

At every step the environment (`mat/envs/rrm/RRM_env.py`):

1. strips the padding and reorders the action into `[all association scores] + [all RB scores]`;
2. normalizes it to `[0, 1]`;
3. optionally refines it with local search (`mat/utils/local_search.py`);
4. projects it onto the feasible set (unique association, $N_b$ users per base station, $N_{rb}$ RBs per user);
5. computes the shared reward from the current channel (`MARLstep_withCurrentH` in `marlcustomeEnv.py`).

## Repository structure

```
.
├── RRM.py                    # network model: base stations, users, path loss (3GPP), SINR
├── marlcustomeEnv.py         # MarlCustomEnv4: multi-agent wrapper and rate computation
├── SNR/                      # heuristic MAX-SNR baseline
│   ├── MAX-SNR.md            # problem formulation and algorithm
│   ├── MAX_SNR.py            # generate a scenario and run MAX-SNR
│   ├── module/               # standalone gymnasium environment and utilities
│   ├── env/7BS84UE20RB/      # saved scenario instance
│   └── *.yaml                # scenario settings
└── mat/                      # Multi-Agent Transformer (adapted from PKU-MARL)
    ├── algorithms/mat/       # encoder, decoder, MAT/GRU variants, policy, PPO trainer
    ├── envs/
    │   ├── rrm/RRM_env.py    # RRM environment used for training
    │   └── starcraft2, football, ma_mujoco, mpe, dexteroushandenvs   # original benchmarks
    ├── runner/shared/        # one runner per environment (rrm_runner.py, ...)
    ├── utils/                # buffer, value normalization, local search
    ├── config.py             # all training arguments
    └── scripts/
        ├── train_rrm.sh      # entry point for the RRM experiments
        ├── train_*.sh        # entry points for the original benchmarks
        ├── train/            # train_rrm.py, train_smac.py, ...
        ├── config_environment_setting_MAT.yaml
        └── results/RRM/mat/  # training logs and checkpoints
```

## Getting started

The code needs Python 3, PyTorch, `gym`/`gymnasium`, `numpy`, `pyyaml` and
`wandb`. The original benchmarks (SMAC, Google Research Football, MuJoCo,
Bi-DexHands) need their own simulators. See the
[PKU MAT README](https://github.com/PKU-MARL/Multi-Agent-Transformer) for
how to install them.

Train MAT on the RRM environment:

```bash
cd mat/scripts
bash train_rrm.sh
```

which runs

```bash
python train/train_rrm.py --env_name RRM --algorithm_name mat \
    --experiment_name rrm_test --seed 1 \
    --n_rollout_threads 1 --episode_length 25 --num_env_steps 10000000 \
    --ppo_epoch 10 --clip_param 0.05 --lr 5e-4 --critic_lr 5e-4 --n_block 1 \
    --use_eval --eval_interval 10 \
    --config_env config_environment_setting_MAT.yaml
```

Logs are written to `mat/scripts/results/RRM/mat/<experiment_name>/` and, if
enabled, to Weights & Biases.

Optional local search on top of the policy's actions:

| Flag | Effect |
|---|---|
| `--use_local_search` | turn local search on |
| `--ls_budget N` | number of candidate evaluations per step |
| `--ls_k K` | association entries flipped per candidate |
| `--ls_stepwise` | greedy stepwise search instead of random sampling |
| `--ls_include_rb` | also search RB entries (with `--ls_stepwise`) |
| `--ls_interval N` | apply local search every N steps |

Run the MAX-SNR baseline:

```bash
cd SNR
python MAX_SNR.py
```

## Scenario configuration

The network is defined in `mat/scripts/config_environment_setting_MAT.yaml`.
Default values:

| Key | Value | Meaning |
|---|---|---|
| `nUEs` | 84 | number of users |
| `nMBS` | 7 | number of macro base stations (agents) |
| `nRBs` | 40 | number of resource blocks |
| `Nb` | 20 | max users per base station |
| `Nrb` | 10 | max RBs per user |
| `txpowerMBSdBm` | 50 | macro base station transmit power (dBm) |
| `BW` | 180000 | bandwidth per RB (Hz) |
| `N0` | -174 | noise power spectral density (dBm/Hz) |
| `fc` | 2.5 | carrier frequency (GHz) |
| `x_max`, `y_max` | 2500 | area size (m) |

Pico and femto base stations (`nPBS`, `nFBS`) are supported by the network
model but are off in the default scenario.

## Other environments

The original MAT benchmarks are kept and can be run with their scripts in
`mat/scripts/`: `train_smac.sh`, `train_smac_multi.sh`,
`train_smac_few_shot.sh`, `train_football.sh`, `train_mujoco.sh`,
`train_mpe.sh` and `train_hands.sh`.

## Acknowledgements

- Multi-Agent Transformer: Wen, Kuba, Lin, Zhang, Wen, Wang, Yang.
  *Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.*
  NeurIPS 2022. [Code](https://github.com/PKU-MARL/Multi-Agent-Transformer).
- The standalone environment in `SNR/module/environment.py` was written by
  Jiansheng Li (2023).
