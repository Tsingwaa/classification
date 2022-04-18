export PYTHONPATH=$HOME/Projects/classification

# Distributed Training without amp
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train.py --config_path $3

# Single GPU
# CUDA_VISIBLE_DEVICES="$2" python3 train.py --config_path "$1" # --lr "$3" --wd "$4"
# CUDA_VISIBLE_DEVICES=8 python3 train.py --config_path configs/Xray9/r50pre_CE.yaml --seed 1
# CUDA_VISIBLE_DEVICES=8 python3 train.py --config_path configs/Xray9/r50pre_CE.yaml --seed 2

# CUDA_VISIBLE_DEVICES=8 python3 train.py --config_path configs/Xray9/r50pre_CE_DRW.yaml --seed 1
# CUDA_VISIBLE_DEVICES=8 python3 train.py --config_path configs/Xray9/r50pre_CE_DRW.yaml --seed 2

CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_RS.yaml --seed 1
CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_RS.yaml --seed 2

CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_RW.yaml --seed 1
CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_RW.yaml --seed 2

CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_Focal.yaml --seed 1
CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_Focal.yaml --seed 2

CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_CB-Focal.yaml --seed 1
CUDA_VISIBLE_DEVICES=6 python3 train.py --config_path configs/Xray9/r50pre_CB-Focal.yaml --seed 2

# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path configs/PathMNIST/r50_CE.yaml
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path configs/PathMNIST/r50_CE_DRW.yaml
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path configs/PathMNIST/r50_RS.yaml
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path configs/PathMNIST/r50_RW.yaml
