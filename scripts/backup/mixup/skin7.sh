export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Skin7/ds121pre_mixup_remix.yaml'

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Skin7/ds121pre_mixup.yaml'

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Skin7/mbv2pre_mixup_remix.yaml'

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Skin7/mbv2pre_mixup.yaml'

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Skin7/r50pre_mixup_remix.yaml'

# CUDA_VISIBLE_DEVICES="$1" python3 train.py --local_rank -1 --config_path 'configs/Skin7/r50pre_mixup.yaml'