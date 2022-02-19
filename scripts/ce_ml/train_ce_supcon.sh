export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_ce_supcon.py --config_path $3

# Single-GPU Training
# cd ../ml && sh train_supcon.sh
# cd ../ce_ml/
# CUDA_VISIBLE_DEVICES="$2" python3 train_ce_supcon.py --config_path "$1"  #  --lambda_weight "$3"

##################################################################### SKin7 #####################################################################
# CE+SupContrast
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

# CE+SimCLR
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.005

# RS + SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_RS_supcon.yaml"
# RW + SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_RW_supcon.yaml"
# CEDRW + SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/r50pre_CEDRW_supcon.yaml"  --drw

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

# RW+SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_RW_supcon.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_CEDRW_supcon.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/PathMNIST/r32_RS_supcon.yaml"

# resize224 ==> ResNet50 pretrained ==> CE+SupContrast
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 55000 train_ce_supcon.py \
#     --config_path "configs/PathMNIST/r50pre_CE_supcon.yaml"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 55000 train_ce_supcon.py \
#     --config_path "configs/PathMNIST/r50pre_RW_supcon.yaml"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 55000 train_ce_supcon.py \
#     --config_path "configs/PathMNIST/r50pre_CEDRW_supcon.yaml"

##################################################################### Xray9 #####################################################################
# SupContrast
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 1.0

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.5

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.05

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.01

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.5

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.05

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.01

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.005

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.001

# SimCLR
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 0.5

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 0.1

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 0.05

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 0.01

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 0.005

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port "$3" train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_CE_simclr.yaml"  --lambda_weight 0.001

# RW + SupContrast
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 60000 train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_RW_supcon.yaml"

# RS + SupContrast
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
#     torchrun --nproc_per_node="$2"  --master_port 50000 train_ce_supcon.py \
#     --config_path "configs/Xray9/r50pre_RS_supcon.yaml"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$1" \
    torchrun --nproc_per_node="$2"  --master_port 50000 train_ce_supcon.py \
    --config_path "configs/Xray9/r50pre_CEDRW_supcon.yaml"
