export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_ce_supcon.py --config_path $3

# Single-GPU Training
# cd ../ml && sh train_supcon.sh
# cd ../ce_ml/
# CUDA_VISIBLE_DEVICES="$2" python3 train_ce_supcon.py --config_path "$1"  #  --lambda_weight "$3"

##################################################################### SKin7 #####################################################################
# SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.005

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.001

# SimCLR
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.001

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.005

##################################################################### PathMNIST #####################################################################
# SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.005

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.001

# SimCLR
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.005

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.001

##################################################################### Xray9 #####################################################################
# SupContrast
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 1.0

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.5

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.05

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.01

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.5

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.05

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.01

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.005

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.001

# SimCLR
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.5

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.05

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.01

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.005

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" torchrun --nproc_per_node="$2" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.001
