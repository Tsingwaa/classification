export PYTHONPATH=$HOME/Projects/classification

# Distributed Training without amp
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
#  torchrun --nproc_per_node=$2 --master_port 30000 train.py \
#  --config_path $3

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
# torchrun --nproc_per_node=$2 --master_port 30000 train.py \
# --config_path configs/PathMNIST/r50_CE.yaml

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
# torchrun --nproc_per_node=$2 --master_port 30000 train.py \
# --config_path configs/PathMNIST/r50_CE_DRW.yaml

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
#  torchrun --nproc_per_node=$2 --master_port 30000 train.py \
#  --config_path configs/PathMNIST/r50pre_RS.yaml

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
#  torchrun --nproc_per_node=$2 --master_port 30000 train.py \
#  --config_path configs/PathMNIST/r50pre_RW.yaml

# Single GPU
# CUDA_VISIBLE_DEVICES="$2" python3 train.py --config_path "$1" # --lr "$3" --wd "$4"
# CUDA_VISIBLE_DEVICES="$2" python3 train_ldam.py --config_path "$1"
# CUDA_VISIBLE_DEVICES=0 python3 train_ldam.py --config_path "configs/Skin7/ds121pre_LDAM_DRW.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path "configs/FGVC/r50_CE.yaml"

CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path "configs/Flowers/r50_CE.yaml"
CUDA_VISIBLE_DEVICES=0 python3 finetune.py --config_path "configs/Flowers/r50_CE.yaml"

CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path "configs/Flowers/r50_CE_DRW.yaml"

cd ../backup/cutmix
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1 --config_path "configs/Skin7/ds121pre_cutmix.yaml"
CUDA_VISIBLE_DEVICES=0 python3 finetune.py --local_rank -1 --config_path "configs/Skin7/ds121pre_cutmix.yaml"