export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_focal1.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_rw.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_oversample.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_CB0.99_focal.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_CB0.99.yaml'
