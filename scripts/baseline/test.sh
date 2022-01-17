export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification

# Distributed Testing

CUDA_VISIBLE_DEVICES=1,4 python3 -W ignore -m torch.distributed.launch\
    --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30006 \
    test.py --config_path "configs/ImageNet_LT/rx50_CE.yaml"