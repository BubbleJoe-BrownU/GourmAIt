# NoisyStudentFood101

## Overview
Noisy Student is a self-supervised training technique released by google research team, who trained an EfficientNet that achieves 88.4% top-1 accuracy on ImageNet. In general, Noisy Student consists of four steps:
  1. Train a teacher classifier on labeled data
  2. Produce pseudu labels or hard labels on a larger unlabeled dataset
  3. Train a larger student classifier on the combination of labeled dataset and unlabeled dataset, adding noise to the student model
  4. Go to step 1, using student as the teacher

For more details please refer to [google research's repo] (https://github.com/google-research/noisystudent) and the [Noisy Student] (https://arxiv.org/abs/1911.04252) article.

In this project, we will use the same training logic to train a light-wieght Convolutional Neural Network (ResNet) on the Food101 dataset. After training and evaluating the model, we will then deploy it on a compute-insufficient device (an android phone).
