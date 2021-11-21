export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification/

CUDA_VISIBLE_DEVICES=0 python3 eval.py --local_rank -1 --config_path \
        'configs/miniIN20_0.05_4step/20211121_r18.yaml'
