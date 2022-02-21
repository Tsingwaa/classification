export PYTHONPATH=$HOME/Projects/classification

# Single GPU
# CUDA_VISIBLE_DEVICES=1 python3 train_supcon.py --config_path "configs/PathMNIST/r32_supcon.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 train_supcon.py --config_path "configs/PathMNIST/r32_simclr.yaml"

# Distributed Training
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
    torchrun --nproc_per_node="$2"  --master_port 60000 train_supcon.py \
    --config_path "configs/PathMNIST/r50pre_supcon.yaml"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
    torchrun --nproc_per_node="$2"  --master_port 60000 train_supcon.py \
    --config_path "configs/PathMNIST/r50pre_simclr.yaml"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_supcon.py \
#     --config_path "configs/Xray9/r50pre_supcon.yaml"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_supcon.py \
#     --config_path "configs/Xray9/r50pre_simclr.yaml"
