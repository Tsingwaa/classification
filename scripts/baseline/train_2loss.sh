export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Training
# CUDA_VISIBLE_DEVICES=$1 python3 -W ignore -m torch.distributed.launch\
#     --nproc_per_node=$2 --master_addr 127.0.0.111 --master_port 30000 \
#     train.py --config_path "configs/miniIN20_0.05/r18_h.yaml"

# Single-GPU Training
# Noise
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.1.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.05.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.01.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.005.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.001.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.01_OS.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.005-0.05.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.005-0.05_OS.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_noise0.001-0.01.yaml'

# Baseline
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_mlp.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_OS.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_RS.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_RW.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r34.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_CB0.99.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_CB0.99_focal1.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_focal1.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/r18_focal2.yaml'

CUDA_VISIBLE_DEVICES=$1 python3 train_2loss.py --local_rank -1\
        --config_path 'configs/miniIN20_0.05/r18_ce_center0.003.yaml'

# Augments
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_noTF.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_cj.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_hflip.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_vflip.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_rscrop.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_rot.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_tsl.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_scale.yaml'
# CUDA_VISIBLE_DEVICES=$1 python3 train.py --local_rank -1\
#         --config_path 'configs/miniIN20_0.05/augments/r18_shear.yaml'
