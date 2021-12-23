export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch \
#     --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniImageNet/20211024_resnet18.yaml"

# Single-GPU Training
# CF10_0.01
CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path \
        "configs/CF10_0.01/r32_LinfPGD0.5_$2.yaml"

# CF10
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path \
#         "configs/CF10/r32_LinfPGD0.5_$2.yaml"

# miniIN20_0.05
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1 --config_path \
#         "configs/miniIN20_0.05/r18_LinfPGD_joint0.5.yaml"
