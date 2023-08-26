import torch
import torch.nn as nn
import torch.nn.functional as F



model = ResNet()
optimizer = model.configure_optimizer()


num_epochs = 100



for epoch in range(num_epochs):
    losses = []
    for x, y in tqdm(get_batch('train', batch_size, device)):
        logits = model(x)