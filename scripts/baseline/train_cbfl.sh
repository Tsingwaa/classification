export PYTHONPATH=$HOME/Projects/classification

CUDA_VISIBLE_DEVICES=2 python3 train.py --config_path 'configs/Xray9/r50pre_CB-Focal.yaml'
CUDA_VISIBLE_DEVICES=2 python3 train.py --config_path 'configs/Skin7/r50pre_CB-Focal.yaml'
CUDA_VISIBLE_DEVICES=2 python3 train.py --config_path 'configs/PathMNIST/r32_CB-Focal_0.9.yaml'
