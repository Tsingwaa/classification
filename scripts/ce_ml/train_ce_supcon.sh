export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_ce_supcon.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 train_ce_supcon.py --config_path "$1"  #  --lambda_weight "$3"


# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_ce_supcon.py \
#     --config_path "configs/FGVC/r50pre_CE_supcon.yaml"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_ce_supcon.py \
#     --config_path "configs/FGVC/r50pre_CE_supcon.yaml"
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_ce_supcon.py \
#     --config_path "configs/FGVC/r50pre_CE_simclr.yaml"
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_ce_supcon.py \
#     --config_path "configs/FGVC/r50pre_RW_supcon.yaml"
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_ce_supcon.py \
#     --config_path "configs/FGVC/r50pre_CEDRW_supcon.yaml"
