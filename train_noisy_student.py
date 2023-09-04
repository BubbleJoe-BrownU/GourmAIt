import math
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import wandb
from randaug import RandAugment
from train_utils import get_lr, get_batch, prepare_model, train, load_model
# configs

wandb_log = True
wandb_project = 'noisy-student'
wandb_name = 'teacher-model-is-best-model'

stepwise_unfreeze = True
init_from = 'from_pretrained'
weight_decay = 1e-1


# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]
to_compile = True

batch_size = 128
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 5 # how many steps to warm up for
lr_decay_iters = 50
learning_rate = 5e-4
min_lr = 1e-5

max_epochs = 100
epoch_num = 0
best_val_loss = float('inf')



def prepare_training():
    """
    prepare a series of models for the Noisy Student Training
    """
    model1, optimizer1, epoch_num1, eval_loss1, stepwise_unfreeze1 = prepare_model('resnet18', init_from='from_pretrained', stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir='noisy-student-model1')
    model2, optimizer2, epoch_num2, eval_loss2, stepwise_unfreeze2 = prepare_model('resnet34', init_from='from_pretrained', stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir='noisy-student-model2')
    model3, optimizer3, epoch_num3, eval_loss3, stepwise_unfreeze3 = prepare_model('resnet50', init_from='from_pretrained', stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir='noisy-student-model3')
    model4, optimizer4, epoch_num4, eval_loss4, stepwise_unfreeze4 = prepare_model('resnet50', init_from='from_pretrained', stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir='noisy-student-model4')
    
    models = [model1, model2, model3, model4]
    optimizer_list = [optimizer1, optimizer2, optimizer3, optimizer4]
    epoch_num_list = [epoch_num1, epoch_num2, epoch_num3, epoch_num4]
    eval_loss_list = [eval_loss1, eval_loss2, eval_loss3, eval_loss4]
    stepwise_unfreeze_list = [stepwise_unfreeze1, stepwise_unfreeze2, stepwise_unfreeze3, stepwise_unfreeze4]

    return models, optimizer_list, epoch_num_list, eval_loss_list, stepwise_unfreeze_list



def main():
    """
    Train a resnet model using the Noisy Student Training. The Noisy Student Training contains four steps:
    (1)
    We first finetuned a resnet18, which was pretrained on ImageNet1K dataset, on the Food101 dataset and use it as the initial teacher model
    (2)
    Then we use this teacher resnet18 model to generate pseudo labels for images from Food101
    (3)
    Then we train a larger student model with input noise (data augmentation) and model noise (Dropout and Stochastic Depth).
    (4)
    We use the student as the new teacher model and continue to step 2.
    """
    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_name)
    
    model_name_list = ['resnet18', 'resnet34', 'resnet50', 'resnet50']
    out_dir_list = [f"noisy-student-model{i}" for i in range(1, 5)]
    for out_dir in out_dir_list:
        os.makedirs(out_dir, exist_ok=True)
    batch_size_list = [384, 384, 384, 384]
    epoch_list = [30, 30, 30, 30]
    models, optimizer_list, epoch_num_list, eval_loss_list, stepwise_unfreeze_list = prepare_training()
    
    for i, model in enumerate(models):
        model = models[i]
        out_dir = out_dir_list[i]
        optimizer = optimizer_list[i]
        batch_size = batch_size_list[i]
        max_epochs = epoch_list[i]
        epoch_num = epoch_num_list[i]
        best_val_loss = eval_loss_list[i]
        stepwise_unfreeze = stepwise_unfreeze_list[i]
        if i == 0:
            train(model=model, 
                optimizer=optimizer, 
                epoch_num=epoch_num, 
                best_val_loss=best_val_loss, 
                stepwise_unfreeze=stepwise_unfreeze, 
                max_epochs=max_epochs, 
                warmup_iters=warmup_iters, 
                lr_decay_iters=lr_decay_iters, 
                decay_lr=decay_lr, 
                learning_rate=learning_rate, 
                min_lr=min_lr, 
                out_dir=out_dir, 
                batch_size=batch_size, 
                device=device, 
                wandb_log=wandb_log)
            models[i] = load_model(model_name_list[i], device, to_compile, out_dir)            
        elif i > 0:
            train(model=model, 
                optimizer=optimizer, 
                epoch_num=epoch_num, 
                best_val_loss=best_val_loss, 
                stepwise_unfreeze=stepwise_unfreeze, 
                max_epochs=max_epochs, 
                warmup_iters=warmup_iters, 
                lr_decay_iters=lr_decay_iters, 
                decay_lr=decay_lr, 
                learning_rate=learning_rate, 
                min_lr=min_lr, 
                out_dir=out_dir, 
                batch_size=batch_size, 
                device=device, 
                wandb_log = wandb_log, 
                teacher=models[i-1], 
                pseudo_label='soft')

            models[i] = load_model(model_name_list[i], device, to_compile, out_dir)            
            
if __name__ == '__main__':
    main()
            

            
            

            
            
    
    
