export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -W ignore -m torch.distributed.launch \
#     --nproc_per_node=4 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_file "configs/train_v1.0_20210827.yaml"

# Single-GPU Training
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath \
        'configs/miniImageNet_0.1/20211024_resnet18_oversample.yaml'
