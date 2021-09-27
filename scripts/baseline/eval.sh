export PYTHONPATH=$PYTHONPATH:/home/waa/Projects/classification/

CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath "configs/20210926_baseline.yaml"
