export PYTHONPATH=$PYTHONPATH:$HOME/Projects/classification/

# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path "configs/CIFAR10/20211010_CIFAR10_randaug_m5n2.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path "configs/CIFAR10/20211010_CIFAR10_randaug_m10n2.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path "configs/CIFAR10/20211012_CIFAR10_randaug_m15n2.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_adapt_randaug_m5n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_adapt_randaug_m10n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_adapt_randaug_m15n2.yaml'

# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_reverse_adapt_randaug_m5n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_reverse_adapt_randaug_m10n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_path 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_reverse_adapt_randaug_m15n2.yaml'

CUDA_VISIBLE_DEVICES=0 python3 eval.py --local_rank -1 --config_path \
        'configs/miniImageNet20_0.05/20211102_r18.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --local_rank -1 --config_path \
#         'configs/miniImageNet_0.05/20211024_resnet18_oversample.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --local_rank -1 --config_path \
#         'configs/miniImageNet/20211024_resnet18.yaml'
