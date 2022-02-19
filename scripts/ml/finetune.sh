export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 finetune.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 finetune.py --config_path "$1"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_supcon.yaml"
CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_simclr.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Xray9/r50pre_supcon.yaml"
CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Xray9/r50pre_simclr.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_supcon.yaml"
CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_simclr.yaml"
