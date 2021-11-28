export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniImageNet/20211024_resnet18.yaml"

# Single-GPU Training
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05_3step/20211125_r18_balsample.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05_3step/20211125_r18_oversample.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/20211127_r18_rw_focal.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/20211127_r18.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/20211127_r34.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/20211127_r18_balsample.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/20211127_r18_oversample.yaml'
