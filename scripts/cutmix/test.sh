export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
CUDA_VISIBLE_DEVICES=4,2 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30004 \
    test.py --config_path "configs/ImageNet_LT/rx50_adapt2_remix_v2_12_0.7.yaml"

# CUDA_VISIBLE_DEVICES=4,2 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30001 \
#     test.py --config_path "configs/ImageNet_LT/rx50_cutmix.yaml"

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=$1 python3 finetune.py --local_rank -1 --config_path $2
