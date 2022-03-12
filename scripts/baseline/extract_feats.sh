export PYTHONPATH=$HOME/Projects/classification

# Single GPU
CUDA_VISIBLE_DEVICES="$2" python3 extract_feats.py --config_path "$1"
