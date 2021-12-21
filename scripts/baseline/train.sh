export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=$2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniIN3_0.02/r18_h.yaml"

# Baseline
CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
        --config_path 'configs/CIFAR10_0.01/r32.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/CIFAR10_0.01/r32_OS.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/CIFAR10_0.01/r32_RW.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/CIFAR10_0.01/r32_RS.yaml'
