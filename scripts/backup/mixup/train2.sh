export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification


CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Xray9/r50pre_mixup_remix.yaml'

CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Xray9/r50pre_mixup.yaml'

CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/PathMNIST/r50pre_mixup_remix.yaml'

CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/PathMNIST/r50pre_mixup.yaml'