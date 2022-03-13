export PYTHONPATH=$HOME/Projects/classification

# Single GPU
# CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --config_path $2

CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py \
    --config_path "configs/Skin7/r50pre_CE_CT.yaml" --lambda_weight 0.07
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py \
    --config_path "configs/Skin7/r50pre_CE_TP.yaml" --lambda_weight 0.03
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py \
    --config_path "configs/Skin7/r50pre_CE_supcon.yaml" --lambda_weight 1.0
