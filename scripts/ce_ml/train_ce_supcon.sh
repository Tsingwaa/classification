export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_ce_supcon.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 train_ce_supcon.py --config_path "$1"  #  --lambda_weight "$3"

CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
    --config_path "configs/Skin7/r50pre_CE_supcon_nobn.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Dogs/r50_CE_supcon.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Flowers/r50_CE_supcon.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/mbv2pre_CE_supcon.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py --config_path "configs/Skin7/ds121pre_CE_supcon.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.2
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.3
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.4
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.6
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.7
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.8
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.9
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 2.0

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.006
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.007
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.008
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.009
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.015
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.025
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.035
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.045
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.055
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.065
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.075
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.085
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.09
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.095
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.15
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.25
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.35
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.4
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.45
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.55
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.65
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.75
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.85
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.9
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.95
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 1.0

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.09
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.4
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.9
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 5.0
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_supcon.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 10.0

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

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py --config_path "configs/FGVC/r50pre_CE_CT.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py --config_path "configs/FGVC/r50pre_CE_TP.yaml"
