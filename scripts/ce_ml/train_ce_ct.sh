export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 train_2loss.py --config_path $3

# Single-GPU Training

# weight=0.0005

# while ($weight < 0.05)
# do CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight $weight
# weight=$weight+0.0005
# done

# weight=0.05

# while ($weight<1.05)
# do CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight $weight

# weight=$weight+0.05

# done

# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0001
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0005
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0015
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.002
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0025
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.003
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0035
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.004
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0045
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 1


CUDA_VISIBLE_DEVICES="$1" python3 train_ce_ct.py \
    --config_path "configs/FGVC/r50pre_CE_CT.yaml"
CUDA_VISIBLE_DEVICES="$1" python3 train_ce_tp.py \
    --config_path "configs/FGVC/r50pre_CE_TP.yaml"
