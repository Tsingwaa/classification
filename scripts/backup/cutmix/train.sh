export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 \
#     torchrun --nproc_per_node=$2 --master_port 30000 train.py \
#     --config_path configs/PathMNIST/r50_CE.yaml

# Single-GPU Training
CUDA_VISIBLE_DEVICES="$2" python3 train.py --local_rank -1 --config_path "$1"

