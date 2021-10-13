export PYTHONPATH=$PYTHONPATH:/home/waa/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -W ignore -m torch.distributed.launch \
#     --nproc_per_node=4 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_file "configs/train_v1.0_20210827.yaml"

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m6n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m7n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m8n2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m9n2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m12n2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m13n2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m14n2.yaml'
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_fpath 'configs/CIFAR10/20211010_CIFAR10_randaug_m15n2.yaml'
