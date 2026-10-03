#!/bin/sh
CONFIG=config_environment_setting_MAT.yaml
algo="mat"
exp="rrm_test"
seed=1

echo "using env config $CONFIG, algo=${algo}, exp=${exp}, seed=${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_rrm.py \
    --env_name RRM \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --seed ${seed} \
    --n_training_threads 16 \
    --n_rollout_threads 1 \
    --n_eval_rollout_threads 1 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 10000000 \
    --ppo_epoch 10 \
    --clip_param 0.05 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --use_ReLU \
    --n_block 1 \
    --gain 0.01 \
    --use_eval \
    --eval_interval 10 \
    --config_env ${CONFIG}