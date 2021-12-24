export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=$2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniIN20_0.05/r18_h.yaml"

# Single-GPU Training
CUDA_VISIBLE_DEVICES=$1 python3 finetune.py --local_rank -1\
        --config_path "configs/miniIN20_0.05/r18_LinfPGD0.5_$2.yaml"
