# NoisyStudentFood101

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
- Noisy Student Training
- Toggle between soft and hard pseudo Labels

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
We report the evaluation of models trained using Noisy Student Training with 50 epochs each iteration. For the purpose of comparison, we also trained a resnet50 model directly for 200 epochs. Here we report the performance of its best checkpoint with regard to the test loss.
**Results of Noisy Student Training with soft pseudo labels**
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 0.023 (direct) | 1.096 | 78.18% |
| resnet34 | 50 | 0.223 (soft) | 0.696 | 81.09% |
| resnet50 | 50 | 0.182 (soft) | 0.514 | 85.83% |
| resnet50 | 50 | 0.159 (soft) | 0.523 | 85.84% |

**Results of Noisy Student Training with hard pseudo labels**
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 0.023 (direct) | 1.096 | 78.18% |
| resnet34 | 50 | 0.223 (soft) | 0.696 | 81.09% |
| resnet50 | 50 | 0.182 (soft) | 0.514 | 85.83% |
| resnet50 | 50 | 0.159 (soft) | 0.523 | 85.84% |

**Results of directly training on Food101 as control**
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet50 | 200 | 0.281 (direct) | 0.588 | 84.36% |

Note: in the Train loss entries of the table, we marked out where labels used to calculate the training loss come from. Direct means labels are the default one-hot labels provided by the dataset, soft means labels are the soft pseudo labels (probability distribution) produced by the teacher model, and hard means labels are the hard pseudo labels (one-hot prediction) produced by the teacher model.

