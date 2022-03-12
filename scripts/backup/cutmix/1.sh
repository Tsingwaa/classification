export PYTHONPATH=$PYTHONPATH:$HOME/project/classification

CUDA_VISIBLE_DEVICES=2 python train.py --config_path "configs/Skin7/r50pre_cutmix_remix.yaml"

CUDA_VISIBLE_DEVICES=2 python train.py --config_path "configs/Skin7/ds121pre_cutmix_remix.yaml"

CUDA_VISIBLE_DEVICES=2 python train.py --config_path "configs/Skin7/mbv2pre_cutmix_remix.yaml"

CUDA_VISIBLE_DEVICES=2 python train.py --config_path "configs/Flowers/r50_cutmix_remix.yaml"

CUDA_VISIBLE_DEVICES=2 python train.py --config_path "configs/Dogs_0.1/r50_cutmix_remix.yaml"