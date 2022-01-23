export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training without amp
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train.py --config_path $3

# Single GPU
CUDA_VISIBLE_DEVICES=$2 python3 train.py --config_path $1 --lr $3 --wd $4
