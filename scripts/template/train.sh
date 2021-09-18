export PYTHONPATH=$PYTHONPATH:/home/hadoop-mtcv/cephfs/data/zengchenghua/projects/mt_fgvc/
CUDA_VISIBLE_DEVICES=1,2,3 python3.7 -W ignore -m torch.distributed.launch \
    --nproc_per_node=3 --master_addr 127.0.0.111 --master_port 30000 \
    train.py --config_file "configs/train_v1.0_20210827.yaml"
