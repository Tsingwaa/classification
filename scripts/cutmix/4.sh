export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# sleep 90m
CUDA_VISIBLE_DEVICES=2,4 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30008 \
    train.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_4_0.6.yaml"

CUDA_VISIBLE_DEVICES=2,4 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30008 \
    finetune.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_4_0.6.yaml"

CUDA_VISIBLE_DEVICES=2,4 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30008 \
    test.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_4_0.6.yaml"
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.3.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.3.yaml' --seed 1
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.3.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.4.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.4.yaml' --seed 1
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2_remix_v3_3_0.4.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt1.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt1.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt1.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt2.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt3.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt3.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt3.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt4.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt4.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt4.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt5.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt5.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt5.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt6.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt6.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt6.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt7.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt7.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt7.yaml' --seed 2

# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt8.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt8.yaml' --seed 1 
# CUDA_VISIBLE_DEVICES=9 python3 train.py --local_rank -1 --config_path 'configs/CF100_0.01/r32_cutmix_adapt8.yaml' --seed 2