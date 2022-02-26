export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=2,4 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30001 \
#     finetune.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_2_0.6.yaml"

# CUDA_VISIBLE_DEVICES=2,4 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30001 \
#     finetune.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_4_0.6.yaml"

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=0 python3 finetune.py --local_rank -1 --config_path 'configs/Flowers/r50_cutmix.yaml' --seed 0
CUDA_VISIBLE_DEVICES=1 python3 finetune.py --local_rank -1 --config_path 'configs/Dogs_0.1/r50_cutmix.yaml' --seed 0
# CUDA_VISIBLE_DEVICES=4 python3 finetune.py --local_rank -1 --config_path 'configs/PathMNIST/r32_cutmix.yaml' --seed 0
