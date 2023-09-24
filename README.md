# GourmAIt: Noisy Student Training on Food101

## Background
[Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) is a image classification dataset published in 2014. It contains 101 food categories, with 101,000 images. For each class 750 trainining images are provided as well as 250 manually reviewed test images. The training images contains certain degrees of noise intendedly, which comes in the form of intense colors and wrong labels.

Noisy Student is a self-supervised training technique released by google research team, who trained an EfficientNet that achieves 88.4% top-1 accuracy on ImageNet. In general, Noisy Student consists of four steps:
  1. Train a teacher classifier on labeled data
  2. Produce pseudu labels or hard labels on a larger unlabeled dataset
  3. Train a larger student classifier on the combination of labeled dataset and unlabeled dataset, adding noise to the student model
  4. Go to step 2, using student as the teacher

For more details please refer to [google research's repo](https://github.com/google-research/noisystudent) and the [Noisy Student](https://arxiv.org/abs/1911.04252) article.

## Overview

In this project, we will use the same training logic to train a light-wieght Convolutional Neural Network (ResNet) on the Food101 dataset. After training and evaluating the model, we will then deploy it on a compute-insufficient device (an android phone).

We decided to train a series of resnet18, resnet34, resnet50, resnet50, which we refer to as model1, model2, model3, and model4, respectively, under the Noisy Student Training framework. The first three models are of increasing model sizes, while model3 and model4 are of same model size, both resnet50. This is one major feature of Noisy Student Training that makes it distinctive from knowledge distillation, where the teacher model is of greater capacity than the student model. Following the orignial paper's practice, we applied RandAugment and Stochastic Depth to student models to increase their robustness. We didn't apply Dropout to our models because Dropout loses effectiveness when combined with BatchNorms in resnet.

<p align="center">
<img src="https://drive.google.com/uc?id=1Rq7Ld-qXVziI7w1KkLL7XTeInc7GUkh6" alt="Noisy Student Framework" width="400"/>
</p>

## Features
- Mixed precision training
- Weight decaying AdamW with cosine-decaying learning rate schedule
- Implemented a unified training framework that can be toggled between direct training and the Noisy Student Training
- Adopted ResNet architecture to exhibit [stochstic depth](https://arxiv.org/abs/1603.09382) during training and unfreeze blocks stepwise during training
- Deployed result model on edge devices (in progress)

## Requirements
- Python <= 3.10 (Python > 3.10 is not yet compatible with PyTorch2.0's compile feature)
- pytorch >= 2.0
- pytorch-cuda
- Torchvision

## Usage
### Dataset
Prior to any model operations, download the `food101` dataset by running:
```python
python datasets/prepare_datasets.py
```
Which will create a `food-101` directory inside the `datasets` directory and download the dataset in about 20 mins. The total size of the dataset is 4.65 gigabytes.

### Training model
To train a model directly on food101 dataset:
```python
python train_directly.py \
--dataset_dir datasets \
--out_dir out-train-directly \
--lr_decay \
--learning_rate 1e-3 \
--min_lr 1e-5 \
--init_from scratch
```

To train a model using Noisy Student Training:
```python
python train_nosiy_student.py \
--dataset_dir datasets \
--out_dir out-noisy-student \
--lr_decay \
--learning_rate 1e-3 \
--min_lr 1e-5 \
--init_from from_pretrained \
--pseudo-label soft
```

The choices of customizable hyperparameters are much more abundant than as shown above. Feel free to modify global variables inside script or add more command line argument options.

## Result
We report the evaluation of models trained using Noisy Student Training with 50 epochs each iteration. The whole training process took about 80 hours on a Titan RTX GPU. For the purpose of comparison, we also trained a resnet50 model directly for 200 epochs. Here we report the performance of its best checkpoint with regard to the test loss.

**Results of Noisy Student Training with soft pseudo labels w/o stochstic depth**
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 0.023 (direct) | 1.096 | 78.18% |
| resnet34 | 50 | 0.223 (soft) | 0.696 | 81.09% |
| resnet50 | 50 | 0.182 (soft) | 0.514 | 85.83% |
| resnet50 | 50 | 0.159 (soft) | 0.523 | 85.84% |

**Results of Noisy Student Training with hard pseudo labels w/o stochastic depth**
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 0.023 (direct) | 1.096 | 78.18% |
| resnet34 | 50 | 0.009 (hard) | 0.932 | 81.57% |
| resnet50 | 50 | 0.005 (hard) | 0.510 | 86.27% |
| resnet50 | 50 | 0.015 (hard) | 0.513 | 86.28% |

**Results of Noisy Student Training with soft pseudo labels w/ stochstic depth**
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 1.27 (direct) | 0.711 | 80.35% |
| resnet34 | 50 | 0.223 (soft) | 0.696 | 81.09% |
| resnet50 | 50 | 0.182 (soft) | 0.514 | 85.83% |
| resnet50 | 50 | 0.159 (soft) | 0.523 | 85.84% |

**Results of directly training on Food101 w/ stochastic depth** [training report](https://api.wandb.ai/links/brownu_ai/fg42itt2)
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet50 | 150 | 0.457 (direct) | 0.488 | 86.59% |

Note: in the Train loss entries of the table, we marked out where labels used to calculate the training loss come from. Direct means labels are the default one-hot labels provided by the dataset, soft means labels are the soft pseudo labels (probability distribution) produced by the teacher model, and hard means labels are the hard pseudo labels (one-hot prediction) produced by the teacher model.

From the result of Noisy Student Training, both with soft pseudo label or hard pseudo label, we can see that in the first three iterations, model's performance increases as model's capacity (depth or number of parameters) increases. And during the training, there is **still an slight increasing trend** for resnet34 and resnet50 near epoch 50. It's recommended to **increase the maximum epochs to further gain an improvement of model performance**. And we also noticed that an extra iteration of training for resnet50 (the fourth row) does not gain significant increase of accuracy. That's probably because the two model are of the same capacity or the number of training epochs are not large enough. Because of the ineffectiveness of the last iteration, we decided to use the resnet 50 trained in the third iteration for later use and for comparison with the resnet50 directly trained.  

In order to compare Noisy Student Training with regular supervised training, we directly trained a resnet50 model on Food101 for 150 epochs, which is comparable to the total number of epochs in the first iterations of Noisy Student Training. The performance of resnet50 directly trained is 2% lower than resnet50 trained in the third iteration in Noisy Student Training. And the test loss and accuracy during training is fluctuating because the noise in training labels. Simply put, in a noisy dataset such as Food101, traininig models using pseudo labels generated by a modest performant teacher model is better than using the default labels. Noisy Student Training enables models to be trainined more smoothly on noisy datasets with escalated-capacity model groups. In the original paper, authors used both high quality labels from ImageNet1K and pseudo labels from a heavily noisy dataset, JFT300M, to train a series of EfficientNet of increasing model size and achieved remarkable results, increasing the ImageNet top-1 accuracy to 88.4%. Due to the limits of compute resource and problem setting, we only use one dataset, Food101, to train our model, which only offers pseudo labels for the training. In the furture work, if the initial result of our research is considered to be important in real-world application, we plan to train our model on a larger-scale dataset to further gain improvement.

## citation
Our project is based on the following works. If you find some part of this project useful, please cite the following works!

```
@article{xie2019self,
  title={Self-training with Noisy Student improves ImageNet classification},
  author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
  journal={arXiv preprint arXiv:1911.04252},
  year={2019}
}
```

```
@article{DBLP:journals/corr/HeZRS15,
  author       = {Kaiming He and
                  Xiangyu Zhang and
                  Shaoqing Ren and
                  Jian Sun},
  title        = {Deep Residual Learning for Image Recognition},
  journal      = {CoRR},
  volume       = {abs/1512.03385},
  year         = {2015},
  url          = {http://arxiv.org/abs/1512.03385},
  eprinttype    = {arXiv},
  eprint       = {1512.03385},
  timestamp    = {Wed, 25 Jan 2023 11:01:16 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{DBLP:journals/corr/HuangSLSW16,
  author       = {Gao Huang and
                  Yu Sun and
                  Zhuang Liu and
                  Daniel Sedra and
                  Kilian Q. Weinberger},
  title        = {Deep Networks with Stochastic Depth},
  journal      = {CoRR},
  volume       = {abs/1603.09382},
  year         = {2016},
  url          = {http://arxiv.org/abs/1603.09382},
  eprinttype    = {arXiv},
  eprint       = {1603.09382},
  timestamp    = {Sat, 15 Dec 2018 13:25:43 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/HuangSLSW16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

