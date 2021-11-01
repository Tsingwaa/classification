export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniImageNet/20211024_resnet18.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniImageNet_0.1/20211024_resnet18_oversample.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch \ --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniImageNet_0.05/20211024_resnet18_oversample.yaml"

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
#         --config_path 'configs/miniImageNet20_0.05/20211101_r18.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
        --config_path 'configs/miniImageNet20_0.05/20211101_r18_balsample.yaml'
