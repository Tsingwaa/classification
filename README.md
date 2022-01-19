# General Framework for Image Classification

It is a general framework for image classification. Specifically, our research field is imbalanced image classification, therefore, our framework integrates various imbalanced methods.

## Introduce

* base: all parent wrap module for trainer, dataset, network, etc.
  * base_trainer: BaseTrainer including all basic functionality, e.g. set configs, initilize modules
  * base_sampler: TODO
  * base_dataset: TODO
  * base_module: TODO
  * base_loss: TODO
* data_loader: all module related to loading data
  * data: images lists of various dataset
  * dataset: various Dataset class inherited from torch.utils.data.dataset.Dataset
  * sampler: various Sampler class inherited from torch.utils.data.sampler.Sampler
  * transform: various transforms for data augmentation
* model: all modules related to model
  * network: general backbones, e.g. ResNet, ResNeXt
  * module: special modules, e.g. adversarial attack, mixup/cutmix.
  * loss: various Loss class inherited from torch.nn.Module
* scripts: scripts related to specific experiment
  * various experiment scripts: one script folder for each experiment
    * configs: store all .yaml run configs by dataset
    * *.py: the python script including Trainer inherited from the BaseTrainer
    * *.sh: the shell script corresponding to the python script
* utils: various functional utility
  * metrics: modules about measure the performance
  * compute_mean_std: get mean and std for various dataset to normalize
  * other utility not commonly used

## How to use

1. clone the repository
2. create a new copy of "scripts/baseline" as your experiment folder
3. if special module is required, add the module to the corresponding parent folder (Note: add the new module to Registry dict, i.e. decorate register_module)
4. rewrite the *.py script inherited from BaseTrainer imitating "scripts/baseline/train.py"
5. rewrite the *.sh script imitating "scripts/baseline/train.sh"
6. run the *.sh script

## FAQ

* What is Registry?
  * Kind of dictionary to get class by the key, i.e. class name.
* Single-GPU or Distributed training?
  * Referred to the *.sh script, if single-gpu, manually set the local_rank to -1, if multi-gpu, let the local_rank automatically set by torch.distributed.launch.

## Note

This repo is now not open to the public. But I will systematically optimize and open this framework when my graduation research is finished.

## Acknowledgements

Special thanks to Xiaoping Wu and Shaobo Guo!
