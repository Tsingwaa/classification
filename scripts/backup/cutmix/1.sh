export PYTHONPATH=$PYTHONPATH:$HOME/project/classification

CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 1
CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 2

CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.02/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.02/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 1
CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.02/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 2

CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 1
CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt2_remix_v4_2_0.6.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.7.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.7.yaml' --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.7.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.8.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.8.yaml' --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.8.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.9.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.9.yaml' --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.01/r32_cutmix_adapt2_remix_v2_2_0.9.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt1.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt1.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt1.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt2.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt2.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt2.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt3.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt3.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt3.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt4.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt4.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt4.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt5.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt5.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt5.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt6.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt6.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt6.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt7.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt7.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt7.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt8.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt8.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path 'configs/CF10_0.1/r32_cutmix_adapt8.yaml' --seed 2