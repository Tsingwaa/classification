export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_2loss.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 train_ce_tp.py --config_path "$1" --lambda_weight "$3" --margin "$4"

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/PathMNIST/r50pre_CE_TP_bs64.yaml"

# sh train_ce_ct.sh "$1"

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/Flowers/r50_CE_TP.yaml" --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/Flowers/r50_CE_TP.yaml" --lambda_weight 0.05
CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/Flowers/r50_CE_TP.yaml" --lambda_weight 0.001


# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/Skin7/ds121pre_CE_TP.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/Skin7/mbv2pre_CE_TP.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0001
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0005
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0015
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.002
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0025
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.003
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0035
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.004
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0045
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.05

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.09

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.1

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.4

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.5

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.9

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 1



# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 10
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 15
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 20
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 25
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 30
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 35
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 40
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 45
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 50
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 55
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 60
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 65
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 70
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 75
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 80
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 85
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 90
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 95
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 100

