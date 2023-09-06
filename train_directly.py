import os
import torch
import torch.nn as nn
import argparse
import wandb
from train_utils import prepare_model, train

wandb_log = True
wandb_project = 'noisy-student'
wandb_name = 'train-resnet50-directly'

# default config
out_dir = 'train-resnet50-directly-for-120-epochs'
dataset_dir = 'datasets'
model_name = 'resnet50'
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
lr_decay_iters = 150
learning_rate = 1e-3
min_lr = 1e-5

max_epochs = 150
epoch_num = 0
best_val_loss = float('inf')

            
def main():
    # to overwrite some default settings
    # add more as you would like
    parser = argparse.ArgumentParser(description='maybe you would like to overwrite some default arguments')
    parser.add_argument('--model-name', default='resnet50')
    parser.add_argument('--init-from', default='from_pretrained')
    parser.add_argument('-o', '--out-dir', default='train-resnet50-directly-for-150-epochs')
    parser.add_argument('-l', '--learning-rate', default=1e-3)
    args = parser.parse_args()

    # overwrite some arguments
    model_name = args.model_name
    init_from = args.init_from
    out_dir = args.out_dir
    learning_rate = args.learning_rate

    os.makedirs(out_dir, exist_ok=True)

    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_name)

    model, optimizer, epoch_num, best_val_loss, stepwise_unfreeze = prepare_model(model_name=model_name, 
                                                                                  init_from=init_from, 
                                                                                  stepwise_unfreeze=True, 
                                                                                  device=device, 
                                                                                  to_compile=to_compile, 
                                                                                  weight_decay=weight_decay, 
                                                                                  learning_rate=learning_rate, 
                                                                                  out_dir=out_dir)

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

if __name__ == "__main__":
    main()
