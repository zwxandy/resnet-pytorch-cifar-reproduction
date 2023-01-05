# Reproduce ResNet32 on CIFAR-10 in PyTorch

# How to run

## Training

[timm](https://github.com/rwightman/pytorch-image-models) is recommended for image classification training 
and required for the training script provided in this repository:
### Distributed training
```shell
./dist_classification.sh $NUM_GPUS -c $CONFIG_FILE /path/to/dataset
```

You can use our training configurations provided in `configs/`:
```shell
./dist_classification.sh 8 -c configs/cifar10_resnet32.yml --model resnet32 /path/to/cifar10
```

### Non-distributed training
```shell
python train.py -c configs/datasets/cifar10_resnet32.yml --model resnet32 /path/to/cifar10
```

### Models and config files
We've updated this repository and moved the previous training script and the checkpoints associated 
with it to `examples/`. The new training script here is just the `timm` training script. We've provided
the checkpoints associated with it in the next section, and the hyperparameters are all provided in
`configs/pretrained` for models trained from scratch, and `configs/finetuned` for fine-tuned models.

# Results
Type can be read in the format `L/PxC` where `L` is the number of transformer
layers, `P` is the patch/convolution size, and `C` (CCT only) is the number of
convolutional layers.

# Citation
```bibtex
@article{hassani2021escaping,
	title        = {Escaping the Big Data Paradigm with Compact Transformers},
	author       = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
	year         = 2021,
	url          = {https://arxiv.org/abs/2104.05704},
	eprint       = {2104.05704},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```
