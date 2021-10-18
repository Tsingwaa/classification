export PYTHONPATH=$PYTHONPATH:/home/waa/Projects/classification/

# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath "configs/CIFAR10/20211010_CIFAR10_randaug_m5n2.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath "configs/CIFAR10/20211010_CIFAR10_randaug_m10n2.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath "configs/CIFAR10/20211012_CIFAR10_randaug_m15n2.yaml"
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_adapt_randaug_m5n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_adapt_randaug_m10n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_adapt_randaug_m15n2.yaml'

# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_reverse_adapt_randaug_m5n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_reverse_adapt_randaug_m10n2.yaml'
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath 'configs/CIFAR10_0.01/20211014_CIFAR10_0.01_reverse_adapt_randaug_m15n2.yaml'

CUDA_VISIBLE_DEVICES=0 python3 eval.py --config_fpath \
        'configs/Caltech256-5/20211018_1_resnet18.yaml'
