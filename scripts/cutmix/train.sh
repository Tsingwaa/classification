export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=4 --master_addr 127.0.0.111 --master_port 30002 \
    train.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_4_0.6.yaml"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=4 --master_addr 127.0.0.111 --master_port 30002 \
    train.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_4_0.5.yaml"
# Baseline
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path $2

# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path 'configs/CIFAR10_0.01/r32.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path 'configs/CIFAR10_0.01/r32_RS.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path 'configs/CIFAR10_0.01/r32_RW.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path 'configs/CIFAR10_0.01/r32_OS.yaml'
