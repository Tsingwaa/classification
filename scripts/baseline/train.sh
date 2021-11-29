export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=$2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniIN20_0.05/r18_h.yaml"

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05_3step/r18_CB0.99999.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05_3step/r18_CB0.9.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05_3step/r18_CB0.9_focal1.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05_3step/r18_CB0.99_focal1.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05_3step/r18_CB0.999_focal1.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05_3step/r18_CB0.9999_focal1.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05_3step/r18_CB0.99999_focal1.yaml'
