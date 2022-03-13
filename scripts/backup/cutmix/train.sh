export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
#     torchrun --nproc_per_node=$1 --master_port 30000 train.py \
#     --config_path configs/PathMNIST/r50_CE.yaml

# Single-GPU Traini4
CUDA_VISIBLE_DEVICES="$2" python3 train.py --local_rank -1 --config_path "$1"

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 \
#     --config_path "configs/Skin7/r50pre_smcm_nosp.yaml"
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 \
#     --config_path "configs/Skin7/r50pre_smcm_nowgt.yaml"



# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 1.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 2
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 2.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 3
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 3.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 4
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 4.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 5.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 6
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 6.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 7
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 7.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 8
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 8.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 9
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 9.5
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --kappa 10

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.48
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.46
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.44
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.42
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.38
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.36
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.34
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.32
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.30
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.28
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.26
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.24
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.22
# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1\
#     --config_path configs/Skin7/r50pre_smcm.yaml --mu 0.20
