export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch \
#     --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniImageNet/20211024_resnet18.yaml"

# Single-GPU Training
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_3step/20211127_r18_LinfPGD_joint0.5.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_3step/20211127_r18_LinfPGD_joint0.5_adapt2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_3step/20211127_r18_LinfPGD_joint0.5_adapt.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_3step/20211127_r18_LinfPGD_joint0.8.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_3step/20211127_r18_LinfPGD_joint0.8_adapt2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_3step/20211126_r18_LinfPGD_joint0.8_adapt.yaml'
