export PYTHONPATH=$HOME/Projects/classification

CUDA_VISIBLE_DEVICES=9 python3 train.py --config_path configs/PathMNIST/r50pre_CE.yaml --seed 1
CUDA_VISIBLE_DEVICES=9 python3 train.py --config_path configs/PathMNIST/r50pre_CE.yaml --seed 2

CUDA_VISIBLE_DEVICES=9 python3 train.py --config_path configs/PathMNIST/r50pre_CE_DRW.yaml --seed 1
CUDA_VISIBLE_DEVICES=9 python3 train.py --config_path configs/PathMNIST/r50pre_CE_DRW.yaml --seed 2