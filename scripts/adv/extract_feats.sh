export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_LinfPGD_joint0.5.yaml'
CUDA_VISIBLE_DEVICES=$1 python3 extract_feats.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_LinfPGD_joint0.5_adapt2.yaml'
