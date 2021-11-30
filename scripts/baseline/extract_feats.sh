export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18.yaml'
