export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=$2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniIN20_0.05/r18_h.yaml"

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=$1 python3 finetune.py --local_rank -1 --config_path $2
CUDA_VISIBLE_DEVICES=2 python3 finetune.py --local_rank -1 --config_path 'configs/PathMNIST/r50pre_mixup_bs64.yaml'
# CUDA_VISIBLE_DEVICES=2 python3 finetune.py --local_rank -1 --config_path 'configs/Skin7/r50pre_mixup.yaml'
# CUDA_VISIBLE_DEVICES=2 python3 finetune.py --local_rank -1 --config_path 'configs/PathMNIST/r32_mixup.yaml'