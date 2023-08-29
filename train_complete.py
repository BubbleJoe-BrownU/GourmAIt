import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models.resnet import resnet50, ResNet50_Weights

import math
import numpy as np
from tqdm import tqdm
import os

from randaug import RandAugment

# default config
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

batch_size = 96
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 40_000 
learning_rate = 1e-3
min_lr = 1e-5

max_epochs = 100
epoch_num = 0
best_val_loss = float('inf')

# gradually unfreeze early layers as training proceeds
def unfreeze_model_stepwise(model, epoch_num):
    if epoch_num // 10 == 0:
        for pn, p in model.named_parameters():
            if pn.startswith('fc'):
                p.requires_grad = True
    else:
        layer_to_unfreeze = epoch_num // 10
        if layer_to_unfreeze > 5:
            return
#         if 0 < layer_to_unfreeze and layer_to_unfreeze <= 4:
        for pn, p in model.named_parameters():
            if pn.startswith(f"layer{5-layer_to_unfreeze}"):
                p.requires_grad = True
        if layer_to_unfreeze == 5:
            for p in model.parameters():
                p.requires_grad = True

if init_from == 'scratch':
    print("training from scratch")
    # no need to freeze layers if training from scratch
    stepwise_unfreeze = False
    model = resnet50(num_classes=101)
elif init_from == 'from_pretrained':
    print("warm start from pretrained checkpoint of resnet 50")
    model = resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=101, bias=True)
    # freeze all layers except the prediction head
    # cannot freeze all here, otherwise no grad is needed and backward would error out
    if stepwise_unfreeze:
        for pn, p in model.named_parameters():
            if not pn.startswith('fc'):
                p.required_grad = False
            
elif init_from == 'resume':
    print(f"resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    epoch_num = checkpoint['epoch_num']
    best_val_loss = checkpoint['best_val_loss']
    
    # resume the unfreeze state
    stepwise_unfreeze = checkpoint['stepwise_unfreeze']
    if stepwise_unfreeze:
        for p in model.parameters():
            p.requires_grad = False

        for i in range(epoch_num + 1):
            if i % 10 == 0:
                unfreeze_model_stepwise(model, i)
    
        
    


model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# initialize a GradScaler. If enabled-False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))


if to_compile:
    print("compiling the model ... (will take a few minutes)")
    unoptimized_model = model
    model = torch.compile(model)



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


# dataloader
def get_batch(split, batch_size, device, labeled=True):
    assert split in {'train', 'test'}
    assert batch_size > 0
    data = Food101(root=dataset_dir, split='train', transform=random_augmentation_preprocess) if split == "train" else Food101(root=dataset_dir, split='test', transform=preprocess)

    data_len = len(data)
    indices = torch.randperm(data_len)


    for i in range(0, data_len, batch_size):
        end = min(data_len, i+batch_size)
        x = torch.stack([data[idx][0] for idx in indices[i:end]]).to(device)
        if labeled:
            y = torch.Tensor([data[idx][1] for idx in indices[i:end]]).to(torch.long).to(device)
            yield x, y
        else:
            yield x

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


    
iter_num = 0 # only used to schedule learning rate

while True:
    if epoch_num % 10 == 0 and stepwise_unfreeze and init_from=='from_prertained':
        unfreeze_model_stepwise(model, epoch_num)
    losses = []
    # training
    model.train()
    for x, y in tqdm(get_batch('train', batch_size, device)):
        # get learning rate from lr_scheduler
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        iter_num += 1
        
        # mixed precision training
        with torch.amp.autocast(device_type=device, dtype=ptdtype):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
#         loss.backward()
#         optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    print(f"epoch {epoch_num}, average training loss: {sum(losses)/len(losses):.4f}")
    
    # evaluating
    model.eval()
    losses = []
    num_correct = 0
    for x, y in tqdm(get_batch('test', batch_size, device)):
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            prediction = torch.argmax(logits, dim=-1)
            num_correct += (prediction == y).sum().item()
        losses.append(loss.item())
    val_loss = np.round(sum(losses)/len(losses), 3)
    val_acc = np.round(num_correct / 25_250, 4)
    print(f"             , average validation loss: {val_loss}, accuracy: {val_acc*100:.2f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if epoch_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_num': epoch_num,
                'best_val_loss': best_val_loss,
                'val_acc': val_acc,
                'stepwise_unfreeze': stepwise_unfreeze
            }
            print(f"validation loss lower than best val loss, saving model checkpoint...")
#             torch.save(checkpoint, os.path.join(out_dir, f"checkpoint-{best_val_loss}.pt"))
            torch.save(checkpoint, os.path.join(out_dir, f"checkpoint.pt"))
    epoch_num += 1
    
    if epoch_num > max_epochs:
        break
            
