import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import inspect
from torchvision import transforms
from torchvision.datasets import Food101
from resnet_with_stochastic_depth import resnet18, resnet34, resnet50
# from torchvision.models.resnet import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import wandb
from randaug import RandAugment

dataset_dir = 'datasets'
# dataset_dir = '../Food101'



def unfreeze_model_stepwise(model, epoch_num, interval=10):
    """
    Gradually unfreeze early layers as training proceeds. 
    In this project, models initialized from scratch are not freezed at all. 
    Models initialized from pretrained ResNet checkpoints, except for the prediction linear head, are freezed (.requires_grad set to False) at first.
    As the training proceeds, layers are unfreezed gradually, with latter layers unfreezed earlier. 
    The speed of unfreezing layers is controlled by the interval parameter, which is set to 10 epochs by default
    """
    if epoch_num // interval != 0:
        return
    if epoch_num // interval == 0:
        for pn, p in model.named_parameters():
            if pn.startswith('fc'):
                p.requires_grad = True
    else:
        layer_to_unfreeze = epoch_num // interval
        if layer_to_unfreeze > 5:
            return
#         if 0 < layer_to_unfreeze and layer_to_unfreeze <= 4:
        for pn, p in model.named_parameters():
            if pn.startswith(f"layer{5-layer_to_unfreeze}"):
                p.requires_grad = True
        if layer_to_unfreeze == 5:
            for p in model.parameters():
                p.requires_grad = True


def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    """
    Learning rate scheduler. Cosine-decaying the learning rate with warmup steps
    For epochs between 0 and warmup_iters, learning rate are linearly increasing from 0 to learning_rate parameter.
    For epochs between warmup_iters and lr_decay_iters, learning rates are cosine-decaying from learning_rate to min_lr.
    For epochs greater than lr_decay_iters, learning rate is constant min_lr
    """
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

def configure_optimizer(model, weight_decay, learning_rate, device):
    """
    Configure an AdamW optimizer that applies weight decaying to model weights
    Those parameters with number of dimensions greater than 1 are considered to be weight decayied
    Use fused AdamW is it's available
    """
    param_dict = {
        pn: p for pn, p in model.named_parameters()
    }

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

# dataloader
def get_batch(split, batch_size, device, dataset_dir, model=None, pseudo_label=None):
    """
    Load data in batches. This function is designed such that it can load data for normal training and evaluation or for noisy student training.
    For normal training and evaluation, when pseudo_label is None, the dataloader will load images augmented by RandAugment algorithm for training and preprocessed original images for evaluation.
    For noisy student training, when pseudo_label is either hard or soft, the dataloder will load images augmented by RandAugment as inputs, and feed preprocessed original images to the teacher model to generate pseudo labels.
    """
    assert split in {'train', 'test'}
    assert batch_size > 0
    # standard preprocess pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # use RandAugment before preprocess
    random_augmentation_preprocess = transforms.Compose(
        [
            RandAugment(),
            preprocess
        ]
    )

    if pseudo_label is None:
        data = Food101(root=dataset_dir, split='train', transform=random_augmentation_preprocess) if split == "train" else Food101(root=dataset_dir, split='test', transform=preprocess)
        data_len = len(data)
        indices = torch.randperm(data_len)
        for i in range(0, data_len, batch_size):
        # for i in range(0, 2*batch_size, batch_size):
            end = min(data_len, i+batch_size)
            inputs = torch.stack([data[idx][0] for idx in indices[i:end]]).to(device)
            targets = torch.Tensor([data[idx][1] for idx in indices[i:end]]).to(torch.long).to(device)
            # yield inputs, targets
            yield inputs, targets

    else:
        assert pseudo_label in {'soft', 'hard'}
        data = Food101(root=dataset_dir, split='train')
        data_len = len(data)
        indices = torch.randperm(data_len)
        
        model.to(device)
        model.eval()
        for i in range(0, data_len, batch_size):
        # for i in range(0, 2*batch_size, batch_size):
            end = min(data_len, i+batch_size)
            inputs = torch.stack([random_augmentation_preprocess(data[idx][0]) for idx in indices[i:end]]).to(device)
            with torch.no_grad():
                targets = model(torch.stack([preprocess(data[idx][0]) for idx in indices[i:end]]).to(device))
            targets = F.softmax(targets, dim=-1) if pseudo_label == 'soft' else torch.argmax(targets, dim=-1)
            yield inputs, targets

def load_model(model_name, device, to_compile, out_dir):
    """
    Load a model from a directory. User should ensure the model type is compatible with the checkpoint saved in out_dir
    This method is to be called when one wants to use the checkpointed model for inference, e.g. to generate pseudo labels.
    If you want to resume training a model, use prepare_model instead, where epoch_num, optimizer state and stepwise unfreezing state will be resumed.
    """
    model_name = model_name.lower()
    assert model_name in {'resnet18', 'resnet34', 'resnet50'}

    model_class, out_channels = {
            'resnet18': [resnet18, 512],
            'resnet34': [resnet34, 512],
            'resnet50': [resnet50, 2048]
    }[model_name]

    model = model_class(num_classes=101)
    model.to(device)
    checkpoint = torch.load(os.path.join(out_dir, 'checkpoint.pt'), map_location=device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
    model.load_state_dict(state_dict)

    if to_compile:
        model = torch.compile(model)

    return model

def prepare_model(model_name, init_from, stepwise_unfreeze, device, to_compile, weight_decay, learning_rate, out_dir):
    """
    Initialize a model
    """
    model_name = model_name.lower()
    assert model_name in {'resnet18', 'resnet34', 'resnet50'}

    model_class, model_weights, out_channels = {
        'resnet18': [resnet18, ResNet18_Weights.IMAGENET1K_V1, 512],
        'resnet34': [resnet34, ResNet34_Weights.IMAGENET1K_V1, 512],
        'resnet50': [resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048]
    }[model_name]

    epoch_num = 0
    best_val_loss = float('inf')
    

    # initialize the model
    if init_from == 'scratch':
        print(f"Initializing {model_name} training from scratch")
        # no need to freeze layers if training from scratch
        stepwise_unfreeze = False
        model = model_class(num_classes=101)
    elif init_from == 'from_pretrained':
        print(f"Initializing {model_name} with model checkpoint pretrained on ImageNet1K")
        model = model_class(weights = model_weights)
        model.fc = nn.Linear(in_features=out_channels, out_features=101, bias=True)
        # freeze all layers except the prediction head
        # cannot freeze all here, otherwise no grad is needed and backward would error out
        if stepwise_unfreeze:
            for pn, p in model.named_parameters():
                if not pn.startswith('fc'):
                    p.required_grad = False
                
    elif init_from == 'resume':
        
        model = model_class(num_classes=101)
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

        print(f"resuming training from {out_dir}, the loaded checkpoint was saved on epoch {epoch_num} with evaluation loss {best_val_loss}")
        # resume the unfreeze state
        stepwise_unfreeze = checkpoint['stepwise_unfreeze']
        if stepwise_unfreeze:
            for p in model.parameters():
                p.requires_grad = False

            for i in range(epoch_num + 1):
                if i % 10 == 0:
                    unfreeze_model_stepwise(model, i)
    # configure the optimizer
    model.to(device)
    optimizer = configure_optimizer(model, weight_decay, learning_rate, device)
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    if to_compile:
        print("compiling the model ... (will take a few minutes)")
        unoptimized_model = model
        model = torch.compile(model)

    return model, optimizer, epoch_num, best_val_loss, stepwise_unfreeze


# we try to keep the structure of training function the same for both supervised training (ST) and noisy student training (NST)
# the only difference is for NST, we provide two extra arguments, teacher and pseudo_label, to the train function
def train(model, optimizer, epoch_num, best_val_loss, stepwise_unfreeze, max_epochs, warmup_iters, lr_decay_iters, decay_lr, learning_rate, min_lr, out_dir, batch_size, device, wandb_log, teacher=None, pseudo_label=None):
    """
    I know the number of parameters passed here are outrageously enormous, but I have little to do with it.
    Sometimes to wrap everything just to make the program looks good, one has to sacrifice something else.
    Try to think of parameters as groups
    First group: model, optimizer, epoch_num, best_val_loss, stepwise_unfreeze, are get from prepare_model;
    Second group: max_epochs, warmup_iters, lr_decay_iters, decay_lr, learning_rate, min_lr, are used to regulate training progress and manipulatie learning rate
    Third group: batch_size, device, teacher=None, pseudo_label, are used to load data with regard to our training framwork
    """

    
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
    }[dtype]
    # initialize a GradScaler. If enabled-False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))
    grad_clip = 1.0

    while True:
        
        
        unfreeze_model_stepwise(model, epoch_num)
        
        # apply learning rate scheduler
        # pass the epoch_num + 1, s.t. epoch_num ranges from 1 to max_epochs
        lr = get_lr(epoch_num+1, learning_rate, min_lr, warmup_iters, lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # training
        losses = []
        model.train()
        for x, y in tqdm(get_batch('train', batch_size, device, dataset_dir, teacher, pseudo_label)):
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
            if len(losses) == 5:
                break
        train_loss = np.round(sum(losses)/len(losses), 3)
        print(f"epoch {epoch_num}, average training loss: {train_loss}")
        
        # evaluating
        model.eval()
        losses = []
        num_correct = 0
        total_pred = 101*250
        for x, y in tqdm(get_batch('test', batch_size, device, dataset_dir)):
            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                prediction = torch.argmax(logits, dim=-1)
                num_correct += (prediction == y).sum().item()
            losses.append(loss.item())
            if len(losses) == 5:
                break
        val_loss = np.round(sum(losses)/len(losses), 3)
        val_acc = np.round(num_correct*100 / total_pred, 2)
        print(f"      , average validation loss: {val_loss}, accuracy: {val_acc}%")

        # log info to wandb
        if wandb_log and False:
            wandb.log(
                {
                    'epoch': epoch_num,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'lr': lr
                }
            )
        if val_loss < best_val_loss and epoch_num % 10 == 0:
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
        
        if epoch_num >= max_epochs:
            break
