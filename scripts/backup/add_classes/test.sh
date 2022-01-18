export PYTHONPATH=$PYTHONPATH:$HOME/project/classification

CUDA_VISIBLE_DEVICES=$1 python3 test.py --config_path $2 --local_rank -1 --seed $3