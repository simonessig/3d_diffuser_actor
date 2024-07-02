exp=3d_diffuser_actor

tasks=(
    pick_box
)
data_dir=./data/real/raw/test/
num_episodes=1
gripper_loc_bounds_file=tasks/real_traj_location_bounds.json
use_instruction=0
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=0
checkpoint=train_logs/real_keypose/last.pth
quaternion_format=wxyz

robot_ip=10.10.10.210
arm_port=50051
gripper_port=50052
cam_calib_file=data/real/raw/pick_box/calibration.json

num_ckpts=${#tasks[@]}
for ((i = 0; i < $num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 python online_evaluation_real/evaluate_policy.py \
        --tasks ${tasks[$i]} \
        --checkpoint $checkpoint \
        --diffusion_timesteps 25 \
        --fps_subsampling_factor $fps_subsampling_factor \
        --lang_enhanced $lang_enhanced \
        --relative_action $relative_action \
        --num_history 3 \
        --test_model 3d_diffuser_actor \
        --cameras $cameras \
        --verbose $verbose \
        --action_dim 7 \
        --collision_checking 0 \
        --predict_trajectory 0 \
        --embedding_dim $embedding_dim \
        --rotation_parametrization "6D" \
        --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
        --data_dir $data_dir \
        --num_episodes $num_episodes \
        --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json \
        --use_instruction $use_instruction \
        --instructions instructions/peract/instructions.pkl \
        --variations {0..0} \
        --max_tries $max_tries \
        --max_steps 20 \
        --seed $seed \
        --gripper_loc_bounds_buffer 0.04 \
        --quaternion_format $quaternion_format \
        --interpolation_length $interpolation_length \
        --dense_interpolation 1 \
        --device cuda \
        --robot_ip $robot_ip \
        --arm_port $arm_port \
        --gripper_port $gripper_port \
        --cam_calib_file $cam_calib_file
done
