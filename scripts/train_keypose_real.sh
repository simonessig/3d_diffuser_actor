main_dir=real_keypose

wandb_project=pdf_task_1

dataset=./data/real/packaged/train
valset=./data/real/packaged/test

use_instruction=1
instructions=instructions/real/pick_box/instructions.pkl

cameras=front

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=2
diffusion_timesteps=100
B=8
C=120
backbone=clip
image_size="256,256"
fps_subsampling_factor=3
gripper_loc_bounds=./tasks/real_loc_bounds.json
gripper_buffer=0.0
quaternion_format=wxyz

train_iters=60000
val_freq=600

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 --master_port $RANDOM \
    main_trajectory_real.py \
    --tasks pick_box \
    --dataset $dataset \
    --valset $valset \
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --image_size $image_size \
    --num_workers 1 \
    --max_episode_length 3 \
    --train_iters $train_iters \
    --embedding_dim $C \
    --use_instruction $use_instruction \
    --instructions $instructions \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq $val_freq \
    --val_iters 6 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --keypose_only 1 \
    --variations {0..0} \
    --lr $lr \
    --num_history $num_history \
    --max_episodes_per_task -1 \
    --relative_action 0 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced 0 \
    --quaternion_format $quaternion_format \
    --wandb_project=$wandb_project \
    --cameras $cameras
