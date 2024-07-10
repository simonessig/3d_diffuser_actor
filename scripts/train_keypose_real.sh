main_dir=real_keypose

wandb_project=pdf_task_0
wandb_group=keypose

dataset=./data/real/packaged/train
valset=./data/real/packaged/test

cameras=front

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=3
diffusion_timesteps=25
B=7
C=192
ngpus=1
backbone=clip
image_size="128,128"
relative_action=0
fps_subsampling_factor=3
lang_enhanced=0
gripper_buffer=0.01
quaternion_format=wxyz

train_iters=30000
val_freq=300

run_log_dir=diffusion_task_real-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-backbone$backbone-S$image_size-R$relative_action-wd$wd
export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory_real.py \
    --tasks pick_box \
    --backbone $backbone \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/real/ \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --image_size $image_size \
    --num_workers 2 \
    --max_episode_length 10 \
    --train_iters $train_iters \
    --embedding_dim $C \
    --use_instruction 0 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq $val_freq \
    --val_iters 2 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 0 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..0} \
    --lr $lr --wd $wd \
    --num_history $num_history \
    --max_episodes_per_task -1 \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --quaternion_format $quaternion_format \
    --run_log_dir $run_log_dir \
    --wandb_project=$wandb_project \
    --wandb_group=$wandb_group
# --cameras $cameras
