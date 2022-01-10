export PYTHONPATH=$PYTHONPATH:$HOME/project/classification

CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2_remix_v2_2_0.5.yaml' --seed 0
CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2_remix_v2_2_0.5.yaml' --seed 1
CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2_remix_v2_2_0.5.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt1_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt1_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt1_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt3_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt3_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt3_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt4_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt4_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt4_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt5_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt5_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt5_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt6_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt6_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt6_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt7_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt7_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt7_remix0.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt8_remix0.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt8_remix0.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=4 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt8_remix0.yaml' --seed 2