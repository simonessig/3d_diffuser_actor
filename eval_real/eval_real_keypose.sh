main_dir=real_keypose

use_instruction=1
instructions=instructions/real/pick_box/instructions.pkl

dense_interpolation=1
interpolation_length=5
num_history=3
diffusion_timesteps=100
embedding_dim=120
cameras="front"
image_size="256,256"
fps_subsampling_factor=3
gripper_loc_bounds=./tasks/real_loc_bounds.json
gripper_buffer=0.0

data_dir=./data/real
num_episodes=1
max_tries=2
verbose=0
seed=0
max_steps=4

checkpoint=train_logs/$main_dir/good-galaxy-15/last.pth

robot_ip=10.10.10.210
arm_port=50051
gripper_port=50052

CUDA_LAUNCH_BLOCKING=1 python \
    eval_real/evaluate_policy.py \
    --tasks pick_box \
    --checkpoint $checkpoint \
    --diffusion_timesteps $diffusion_timesteps \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced 0 \
    --relative_action 0 \
    --num_history $num_history \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --image_size $image_size \
    --verbose $verbose \
    --action_dim 7 \
    --collision_checking 0 \
    --predict_trajectory 0 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$main_dir/seed$seed/${tasks[$i]}.json \
    --use_instruction $use_instruction \
    --instructions $instructions \
    --variations {0..0} \
    --max_tries $max_tries \
    --max_steps $max_steps \
    --seed $seed \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --quaternion_format wxyz \
    --interpolation_length $interpolation_length \
    --dense_interpolation $dense_interpolation \
    --device cuda \
    --robot_ip $robot_ip \
    --arm_port $arm_port \
    --gripper_port $gripper_port
