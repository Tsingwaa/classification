export PYTHONPATH=$PYTHONPATH:/home/hadoop-mtcv/cephfs/data/zengchenghua/projects/mt_fgvc/
CUDA_VISIBLE_DEVICES=1 python3.7 -W ignore val.py \
    --config_file "configs/val_v3.0_20210819_FixMatch.yaml"
