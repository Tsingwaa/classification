export PYTHONPATH=$HOME/Projects/classification

# Distributed Training without amp
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$2" torchrun --nproc_per_node="$3" \
#     --master_addr 127.0.0.1 --master_port 30000 train_supcon.py --config_path "$1"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_supcon.py \
#     --config_path "configs/Skin7/r50pre_supcon.yaml" --out_type "vec_norm"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_supcon.py \
#     --config_path "configs/Skin7/r50pre_supcon.yaml" --out_type "vec_2lp_norm"

# Single GPU
CUDA_VISIBLE_DEVICES="$2" python3 train_supcon.py --config_path "$1" --out_type "vec_norm"
CUDA_VISIBLE_DEVICES="$2" python3 train_supcon.py --config_path "$1" --out_type "vec_2lp_norm"
