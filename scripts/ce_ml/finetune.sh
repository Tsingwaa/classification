export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 finetune.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 finetune.py --config_path "$1" --lambda_weight "$3"  # --margin "$4"


##################################################################### SKin7 #####################################################################
# SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.005

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml"  --lambda_weight 0.001

# SimCLR
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 1.0 --t 1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_simclr.yaml"  --lambda_weight 0.001

##################################################################### PathMNIST #####################################################################
# SupContrast
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.005

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_supcon.yaml"  --lambda_weight 0.001

# SimCLR
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.005

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/PathMNIST/r32_CE_simclr.yaml"  --lambda_weight 0.001

##################################################################### Xray9 #####################################################################
# SupContrast
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 1.0
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.01

# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_supcon.yaml" --lambda_weight 0.001

# SimCLR
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 1
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" finetune.py -config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" finetune.py --config_path "configs/Xray9/r50pre_CE_simclr.yaml" --lambda_weight 0.001
