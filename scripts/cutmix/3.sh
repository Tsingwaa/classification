export PYTHONPATH=$PYTHONPATH:$HOME/project/classification

CUDA_VISIBLE_DEVICES=8 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.1.yaml' --seed 0
CUDA_VISIBLE_DEVICES=8 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.1.yaml' --seed 1
CUDA_VISIBLE_DEVICES=8 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.1.yaml' --seed 2

CUDA_VISIBLE_DEVICES=8 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.2.yaml' --seed 0
CUDA_VISIBLE_DEVICES=8 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.2.yaml' --seed 1
CUDA_VISIBLE_DEVICES=8 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.2.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt1.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt1.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt1.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt2.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt3.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt3.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt3.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt4.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt4.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt4.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt5.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt5.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt5.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt6.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt6.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt6.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt7.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt7.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt7.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt8.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt8.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=6 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.1/r32_cutmix_adapt8.yaml' --seed 2