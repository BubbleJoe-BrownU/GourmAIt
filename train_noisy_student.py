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

from randaug import RandAugment

# configs
out_dir = 'out-resnet50-pretrained-stepwise-unfreeze'
dataset_dir = 'datasets'

rand_aug = True
stepwise_unfreeze = True
init_from = 'from_pretrained'



# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]
to_compile = True

os.makedirs(out_dir, exist_ok=True)

batch_size = 128
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 5 # how many steps to warm up for
lr_decay_iters = 50
learning_rate = 1e-3
min_lr = 1e-5

max_epochs = 100
epoch_num = 0
best_val_loss = float('inf')

def prepare_training():
    model1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model1.fc = nn.Linear(in_features=512, out_features=101, bias=True)
    model2 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V2)
    model2.fc = nn.Linear(in_features=1024, out_features=101, bias=True)
    model3 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model3.fc = nn.Linear(in_features=2048, out_features=101, bias=True)
    model4 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model4.fc = nn.Linear(in_features=204, out_features=101, bias=True)
    
    models = [model1, model2, model3, model4]
    batch_size_list = [128, 128, 96, 64]
    epoch_list = [100, 50, 50, 50]
    return models, batch_size_list, epoch_list

# preprocess the input
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# random augmentation
random_augmentation_preprocess = transforms.Compose(
    [
        RandAugment(),
        preprocess
    ]
)

def get_batch_with_pseudo_label(batch_size, teacher, device, label_model='soft'):
    data = Food101(root=dataset_dir, split='train')
    
    data_len = len(data)
    indices = torch.randperm(data_len)
    
    teacher.to(device)
    teacher.eval()
    
    for i in range(0, data_len, batch_size):
        end = min(data_len, i+batch_size)
        x = torch.stack([random_augmentation_preprocess(data[idx][0]) for idx in indices[i:end]]).to(device)
        with torch.no_grad():
            y = teacher(torch.stack([preprocess(data[idx][0]) for idx in indices[i:end]]).to(device))
        y = F.softmax(y, dim=-1) if label_model == 'soft' else torch.argmax(y, dim=-1)
        yield x, y

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
    
    models, batch_size_list, epoch_list = prepare_training()
    
    for i, model in enumerate(models):
        if i == 0:
#             train_directly(model)
            # would take several hours even days to converge on cheap GPUs
            # laoding checkpointed state dict recommended
            checkpoint_path = 'out-'
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
        elif i > 0:
            batch_size = batch_size_list[i]
            max_epochs = epoch_list[i]
            dataloader = get_batch_with_pseudo_label(batch_size, models[i-1], device)
            
            
    
    