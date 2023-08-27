import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models.resnet import resnet50

import numpy as np
from tqdm import tqdm
import os

# default config
out_dir = 'out-resnet50-pretrained-linear-probe'
dataset_dir = 'datasets'

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]
to_compile = True

batch_size = 48
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 5
lr_decay_epochs = 100
min_lr = 1e-5

model = resnet50()
model.fc = nn.Linear(in_features=2048, out_features=101, bias=True)
# freeze all layers except the prediction headS
# do some linear probe here
for pn, p in model.named_parameters():
    if not pn.startswith('fc'):
        p.required_grad = False
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# initialize a GradScaler. If enabled-False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))


if to_compile:
    print("compiling the model ... (will take a few minutes)")
    unoptimized_model = model
    model = torch.compile(model)

num_epochs = 100
best_val_loss = float('inf')

# preprocess the input
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataloader
def get_batch(split, batch_size, device, labeled=True):
    assert split in {'train', 'test'}
    assert batch_size > 0
    data = Food101(root=dataset_dir, split='train', transform=preprocess) if split == "train" else Food101(root=dataset_dir, split='test', transform=preprocess)

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



for epoch in range(num_epochs):
    losses = []
    # training
    model.train()
    for x, y in tqdm(get_batch('train', batch_size, device)):
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
    print(f"epoch {epoch}, average training loss: {sum(losses)/len(losses):.4f}")
    
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
    print(f"             , average validation loss: {val_loss}, accuracy: {val_acc*100}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if epoch > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_acc': val_acc
            }
            print(f"validation loss lower than best val loss, saving model checkpoint...")
            torch.save(checkpoint, os.path.join(out_dir, f"checkpoint-{best_val_loss}.pt"))
            