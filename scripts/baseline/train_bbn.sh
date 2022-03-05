export PYTHONPATH=$HOME/Projects/classification

# Distributed Training without amp
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train.py --config_path $3

# CUDA_VISIBLE_DEVICES=3,4 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30002 \
#     train_bbn.py --config_path "configs/Xray9/r50pre_BBN.yaml"

# Single GPU
CUDA_VISIBLE_DEVICES="$2" python3 train_bbn.py --config_path "$1" # --lr "$3" --wd "$4"
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path configs/PathMNIST/r32_RS.yaml
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path configs/PathMNIST/r32_RW.yaml
