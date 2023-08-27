import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models.resnet import resnet50

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
to_compile = True
batch_size = 48
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

model = resnet50()
model.fc = nn.Linear(in_features=2048, out_features=101, bias=True)
# freeze all layers except the prediction headS
# do some linear probe here
for pn, p in model.named_parameters():
    if not pn.startswith('fc'):
        p.required_grad = False
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
    data = Food101(root='datasets', split='train', transform=preprocess) if split == "train" else Food101(root='datasets', split='test', transform=preprocess)

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
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch {epoch}, average training loss: {sum(losses)/len(losses):.4f}")
    
    # evaluating
    model.eval()
    losses = []
    for x, y in tqdm(get_batch('test', batch_size, device)):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
    val_loss = sum(losses)/len(losses)
    print(f"             , average validation loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        print(f"validation loss lower than best val loss, saving model checkpoint...")