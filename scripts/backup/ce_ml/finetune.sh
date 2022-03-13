export PYTHONPATH=$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
#     --master_addr 127.0.0.1 --master_port 30000 finetune.py --config_path $3

# Single-GPU Training
# CUDA_VISIBLE_DEVICES="$2" python3 finetune.py --config_path "$1" --lambda_weight "$3"  # --margin "$4"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.00005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.00001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.000005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.00005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.00001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.000005

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Dogs/r50_CE_TP.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Dogs/r50_CE_CT.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Dogs/r50_CE_supcon.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Flowers/r50_CE_TP.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Flowers/r50_CE_CT.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Flowers/r50_CE_supcon.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/mbv2pre_CE_CT.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/ds121pre_CE_CT.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/mbv2pre_CE_supcon.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/ds121pre_CE_supcon.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/ds121pre_CE_TP.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/Skin7/mbv2pre_CE_TP.yaml"

##################################################################### FGVC #####################################################################
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/FGVC/r50pre_CE_TP.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py --config_path "configs/FGVC/r50pre_CE_CT.yaml"

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0015
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.002
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0025
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.003
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0035
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.004
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.0045
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.05

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.09

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.1

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.4

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.5

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 0.9

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_CT.yaml"  --lambda_weight 1


# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0015
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.002
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0025
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.003
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0035
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.004
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.0045
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.05

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.09

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.1

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.4


# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.5

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.9

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 1


# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 10
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 15
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 20
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 25
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 30
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 35
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 40
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 45
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 50
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 55
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 60
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 65
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 70
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 75
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 80
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 85
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 90
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 95
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_TP.yaml"  --lambda_weight 0.001 --margin 100


# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.006
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.007
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.008
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.009
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.015
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.025
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.035
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.045
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.055
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.065
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.075
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.085
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.09
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.095
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.15
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.25
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.35
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.4
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.45
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.55
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.65
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.75
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.85
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.9
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 0.95
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0 --t 1.0

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.001
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.005
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.01
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.02
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.03
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.04
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.05
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.06
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.07
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.08
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.09
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.2
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.3
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.4
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.6
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.7
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.8
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 0.9
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0

# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.1
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.2
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.3
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.4
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.5
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.6
CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
    --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.7
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.8
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.9
# CUDA_VISIBLE_DEVICES="$1" python3 finetune.py \
#     --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 2.0

