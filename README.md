# Reproduce ResNet on CIFAR-10 in PyTorch

As we know, the official ResNet code (including timm lib) only provides models for ImageNet dataset with 4 stages. 
However, in the original ResNet paper (experiments part), **only 3 stages are needed for CIFAR-10 dataset (e.g., # blocks=[5, 5, 5] in ResNet-32)**. 
For example, if we directly run 4-stage ResNet-32 on CIFAR-10, we unfortunately get ~88% top-1 accuracy (but 92.49% in the original ResNet paper).

So, this project reproduces ResNet models (20/32/44/56/110/1202-layer) on CIFAR-10 with timm lib. Finally, we can get 92.45%~92.61% top-1 accuracy for ResNet-32 which matches the result 92.49%.

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

### Models, config files and detailed training hyper-parameters
- Models are in the "src/resnet.py" file. We can choose ResNet-20/32/44/56/110/1202 on CIFAR-10 by simply modify the training command like: `--model resnet56`.
- Hyper-parameters are in the "config/datasets/cifar10_resnet32.yml" file and "train.py".

|  Hyper-parameter   | Value  |
|  ----  | ----  |
| optimizer | sgd |
| learning scheduler | multistep (decay-milestones=[100, 150]) |
| warmup_epochs | 10 (warmup_lr=0.00001) |
| label smoothing | 0.1 |
| batch size | 128 |
|  weight decay  | 1e-4 |
|  momentum  | 0.9 |
| initial learning rate | 0.1 |
| mean | [0.485, 0.456, 0.406] |
| std  | [0.229, 0.224, 0.225] |
| # epochs | 200 |
| with dropout? | No |
| data augmentation| RandomHorizontalFlip, RandomCrop, RandomErasing |


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

@misc{Idelbayev18a,
  author       = "Yerlan Idelbayev",
  title        = "Proper {ResNet} Implementation for {CIFAR10/CIFAR100} in {PyTorch}",
  howpublished = "\url{https://github.com/akamaster/pytorch_resnet_cifar10}",
  note         = "Accessed: 20xx-xx-xx"
}
```
