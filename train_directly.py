import os
import torch
import torch.nn as nn
import argparse
import wandb
from train_utils import prepare_model, train

# there are some arguments we don't pass via command line
# obviously for sake of simplicity, but if you want
# you can replace these global variables with argparse arguments


wandb_log = True
wandb_project = 'noisy-student'
wandb_name = 'train-resnet50-directly'


to_compile = True

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

grad_clip = 1.0
# learning rate decay settings
warmup_iters = 5 # how many steps to warm up for
lr_decay_iters = 150
learning_rate = 1e-4
min_lr = 1e-5



def main():
    """
    Directly train a resnet model with stochastic depth on the Food101 dataset. This achieves pretty good accuracy, though not as good as resnet trained using Noisy Student Training.
    """
    # to overwrite some default settings
    # add more as you would like
    parser = argparse.ArgumentParser(description='maybe you would like to overwrite some default arguments')
    parser.add_argument('--model-name', type=str, default='resnet50', help="the specific resnet model type you would like to train")
    parser.add_argument('--init-from', type=str, default='from_pretrained', help="initialize the model from scratch, from_pretrained, or resume training")
    parser.add_argument('--out-dir', type=str, default='train-resnet50-directly-for-150-epochs', help='the name of directory where we will save or load model checkpoints')
    parser.add_argument('--decay-lr', type=bool, default=True, help="whether to use the cosine weight decay learning rate scheduler with warmup")
    parser.add_argument('--weight-decay', type=float, default=1e-1, help="the weight decaying coefficient used in training")
    parser.add_argument('--stepwise-unfreeze', type=bool, default=True, help="whether to unfreeze a resnet gradually as training goes on")
    parser.add_argument('--max-epochs', type=int, default=150, help="the maximum number of epochs")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="the (default) highest learning rate used in training")
    parser.add_argument('--min-lr', type=float, default=1e-5, help="the minimum learning rate used in training if learning decay is enabled")
    parser.add_argument('--batch-size', type=int, default=128, help="number of data examples in a mini batch")
    
    args = parser.parse_args()

    # overwrite some arguments
    model_name = args.model_name
    init_from = args.init_from
    out_dir = args.out_dir
    decay_lr = args.decay_lr
    weight_decay = args.weight_decay
    stepwise_unfreeze = args.stepwise_unfreeze
    max_epochs = args.max_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    
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
