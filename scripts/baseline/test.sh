export PYTHONPATH=$HOME/Projects/classification

# Distributed Testing
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=$2 \
    --master_addr 127.0.0.1 --master_port 30000 test.py --config_path $3

# Single GPU
# CUDA_VISIBLE_DEVICES=$1 python3 test.py --config_path $2
