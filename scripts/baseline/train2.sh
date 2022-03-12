export PYTHONPATH=$HOME/Projects/classification

# CUDA_VISIBLE_DEVICES=1 python3 train_ldam.py --config_path "configs/Skin7/mbv2pre_LDAM.yaml"
# CUDA_VISIBLE_DEVICES=1 python3 train.py --config_path "configs/FGVC/r50_CE_DRW.yaml"

# CUDA_VISIBLE_DEVICES=1 python3 train.py --config_path "configs/Skin7/mbv2pre_CE.yaml"
# CUDA_VISIBLE_DEVICES=1 python3 train_ldam.py --config_path "configs/Skin7/mbv2pre_LDAM_DRW.yaml"

CUDA_VISIBLE_DEVICES=1 python3 train_ldam.py --config_path "configs/Flowers/r50_LDAM.yaml"
CUDA_VISIBLE_DEVICES=1 python3 finetune.py --config_path "configs/Flowers/r50_LDAM.yaml"

CUDA_VISIBLE_DEVICES=1 python3 train_ldam.py --config_path "configs/Flowers/r50_LDAM_DRW.yaml"
