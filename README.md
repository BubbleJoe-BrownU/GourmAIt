# GourmAIt: Noisy Student Training on Food101

## Background
[Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) is an image classification dataset published in 2014. It contains 101 food categories, with 101,000 images. For each class 750 training images are provided as well as 250 manually reviewed test images. The training images contain certain degrees of noise, which come in the form of intense colors and wrong labels.

Noisy Student is a self-supervised training technique released by Google Research team, who trained an EfficientNet that achieves 88.4% top-1 accuracy on ImageNet. In general, Noisy Student consists of four steps:
  1. Train a teacher classifier on labeled data
  2. Produce pseudo labels or hard labels on a larger unlabeled dataset
  3. Train a larger student classifier on the combination of labeled dataset and unlabeled dataset, adding noise to the student model
  4. Go to step 2, using the student as the teacher

For more details please refer to [Google Research's repo](https://github.com/google-research/noisystudent) and the [Noisy Student](https://arxiv.org/abs/1911.04252) article.

## Overview

In this project, we will use the same training logic to train a lightweight Convolutional Neural Network (ResNet) on the Food101 dataset. Besides implementing the Noisy Student Training, we also implemented stochastic depth, stepwise unfreezing scheduling, and learning rate scheduling to effectively train ResNet.

We decided to train a series of resnet18, resnet34, resnet50, resnet50, which we refer to as model1, model2, model3, and model4, respectively, under the Noisy Student Training framework. The first three models are of increasing model sizes, while model3 and model4 are of the same model size, which are both resnet50. This is one major feature of Noisy Student Training that makes it distinctive from knowledge distillation, where the teacher model is of greater capacity than the student model. Following the original paper's practice, we applied RandAugment and Stochastic Depth to student models to increase their robustness. We didn't apply Dropout to our models because Dropout loses effectiveness when combined with BatchNorms in ResNet.

<p align="center">
<img src="https://drive.google.com/uc?id=1Rq7Ld-qXVziI7w1KkLL7XTeInc7GUkh6" alt="Noisy Student Framework" width="400"/>
</p>

## Features
- Noisy Student Training
- Weight decaying AdamW with the cosine-decaying learning rate schedule
- A unified training framework that can be toggled between direct training and the Noisy Student Training
- Adopted ResNet architecture to exhibit [stochstic depth](https://arxiv.org/abs/1603.09382) during training and unfreeze blocks stepwise during training


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
To train a model directly on the food101 dataset:
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
We report the evaluation of models trained using Noisy Student Training with 50 epochs each iteration. The whole training process took about 80 hours on a Titan RTX GPU. We also did a thorough ablation study to investigate the benefits of introducing Noisy Student Training and Stochastic Depth. Here we report the performance of its best checkpoint with regard to the test loss.

**Results of Noisy Student Training with soft pseudo labels w/o stochastic depth**
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

**Results of Noisy Student Training with hard pseudo labels w/ stochastic depth** [training report](https://api.wandb.ai/links/brownu_ai/cyupb1tt)
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 1.27 (direct) | 0.711 | 80.35% |
| resnet34 | 50 | 0.457 (hard) | 0.630 | 83.45% |
| resnet50 | 50 | 0.425 (hard) | 0.505 | 86.34% |

**Results of Noisy Student Training with soft pseudo labels w/ stochastic depth** [training report](https://api.wandb.ai/links/brownu_ai/jn0sque3)
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet18 | 50 | 1.27 (direct) | 0.711 | 80.35% |
| resnet34 | 50 | 0.755 (soft) | 0.580 | 83.75% |
| resnet50 | 50 | 0.59 (soft) | 0.461 | 87.07% |

**Results of directly training on Food101 w/ stochastic depth** [training report](https://api.wandb.ai/links/brownu_ai/fg42itt2)
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet50 | 150 | 0.457 (direct) | 0.488 | 86.59% |

**Results of directly training on Food101 w/o stochastic depth** [training report](https://api.wandb.ai/links/brownu_ai/ql531iek)
| Model Name | Training Epochs | Train Loss | Test Loss | Test Acc |
| -- | -- | -- | -- | -- |
| resnet50 | 150 | 0.397 (direct) | 0.573 | 83.89% |

Note: in the Train loss entries of the table, we marked out where labels used to calculate the training loss come from. "Direct" means labels are the default one-hot labels provided by the dataset, "soft" means labels are the soft pseudo labels (probability distribution) produced by the teacher model, and "hard" means labels are the hard pseudo labels (one-hot prediction) produced by the teacher model.

From the result of Noisy Student Training with standard ResNets, either with the soft pseudo label or hard pseudo label, we can see that in the first three iterations, the model's performance increases as the model's capacity (depth or the number of parameters) increases. During the training, there is **still a slight increasing trend** for resnet34 and resnet50 near epoch 50. It's recommended to **increase the maximum epochs to further gain an improvement of model performance**. We also noticed that an extra iteration of training for resnet50 (the fourth row) does not gain a significant increase in accuracy. That's probably because the two model are of the same capacity or the number of training epochs are not large enough. Because of the ineffectiveness of the last iteration, we decided to **only use three iterations** for the Noisy Student Training in later experiments.

We also incorporated stochastic depth in the existing ResNet framework and trained resnets with stochastic depth using Noisy Student Training. The results of NST with stochastic are shown beneath the result of NST with standard ResNets. For ResNets with the same number of convolutional layers, the stochastic depth increases the model's accuracy by 1% ~ 2%, and makes the training process more stable at the same time.

In order to compare Noisy Student Training with regular supervised training, we directly trained a resnet50 model on Food101 for 150 epochs, which is comparable to the total number of epochs in the Noisy Student Training of three iterations. The performance of resnet50 directly trained is 2% lower than resnet50 trained in the third iteration in Noisy Student Training using soft pseudo labels. The resnet50 training with NST using hard pseudo labels achieved 86.34% accuracy, slightly lower than directly training resnet50. For directly trained resnet50, the test loss and accuracy during training fluctuate because of the noise in training labels. Simply put, in a noisy dataset such as Food101, training models using pseudo labels generated by a modest performant teacher model is better than using the default labels. Noisy Student Training enables models to be trained more smoothly on noisy datasets with escalated-capacity model groups. 

In the original paper, authors used both high-quality labels from ImageNet1K and pseudo labels from a heavily noisy dataset, JFT300M, to train a series of EfficientNet of increasing model size and achieved remarkable results, increasing the ImageNet top-1 accuracy to 88.4%. Due to the limits of computing resources and the specific problem setting, we only use one dataset, Food101, to train our model, which only offers pseudo labels for the training. In future work, if the initial result of our research is considered to be important in real-world applications, we plan to train our model on a larger-scale dataset to further gain improvement.

## citation
Our project is based on the works listed below. If you find some part of this project useful, please cite the following works!

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

