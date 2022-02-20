export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_2loss.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 train_ce_tp.py --config_path "$1" --lambda_weight "$3" --margin "$4"

CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/PathMNIST/r50pre_CE_TP_bs64.yaml"

sh train_ce_ct.sh "$1"
