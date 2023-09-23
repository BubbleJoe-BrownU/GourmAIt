import argparse
import os
import torch
import torch.nn as nn
import wandb
from train_utils import prepare_model, train, load_model
# configs

wandb_log = True
wandb_project = 'noisy-student'
wandb_name = 'teacher-model-is-best-model'

stepwise_unfreeze = True
init_from = 'resume'
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




def prepare_training(init_from, stepwise_unfreeze, device, to_compile, weight_decay, learning_rate, out_dir_list):
    """
    prepare a series of models for the Noisy Student Training
    """
    model1, optimizer1, epoch_num1, eval_loss1, stepwise_unfreeze1 = prepare_model('resnet18', init_from=init_from, stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir=out_dir_list[0])
    model2, optimizer2, epoch_num2, eval_loss2, stepwise_unfreeze2 = prepare_model('resnet34', init_from=init_from, stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir=out_dir_list[1])
    model3, optimizer3, epoch_num3, eval_loss3, stepwise_unfreeze3 = prepare_model('resnet50', init_from=init_from, stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir=out_dir_list[2])
    model4, optimizer4, epoch_num4, eval_loss4, stepwise_unfreeze4 = prepare_model('resnet50', init_from=init_from, stepwise_unfreeze=True, device=device, to_compile=to_compile, weight_decay=weight_decay, learning_rate=learning_rate, out_dir=out_dir_list[3])
    
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
    parser = argparse.ArgumentParser(description='maybe you would like to overwrite some default settings like learning rate or pseudo label type.')
    parser.add_argument('--init-from', type=str, default='from_pretrained', help="initialize the model from scratch, from_pretrained, or resume training")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="the (default) highest learning rate used in training")
    parser.add_argument('--decay_lr', type=bool, default=True, help="whether to use the cosine weight decay learning rate scheduler with warmup")
    parser.add_argument('--weight-decay', type=float, default=1e-1, help="the weight decaying coefficient used in training")
    parser.add_argument('--stepwise-unfreeze', type=bool, default=True, help="whether to unfreeze a resnet gradually as training goes on")
    parser.add_argument('--min-lr', type=float, default=1e-5, help="the minimum learning rate used in training if learning decay is enabled")
    parser.add_argument('--pseudo-label', type=str, default='soft', help="whether to use soft pseudo label or hard pseudo label")
    

    args = parser.parse_args()
    
    decay_lr = args.decay_lr
    pseudo_label = args.pseudo_label
    weight_decay = args.weight_decay
    init_from = args.init_from
    learning_rate = args.learning_rate
    min_lr = args.min_lr
    stepwise_unfreeze = args.stepwise_unfreeze

    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_name)
    
    model_name_list = ['resnet18', 'resnet34', 'resnet50', 'resnet50']
    out_dir_list = [f"noisy-student-model{i}" for i in range(1, 5)]
    for out_dir in out_dir_list:
        os.makedirs(out_dir, exist_ok=True)
    batch_size_list = [384, 384, 384, 384]
    epoch_list = [50, 50, 50, 50]
    models, optimizer_list, epoch_num_list, eval_loss_list, stepwise_unfreeze_list = prepare_training(init_from, stepwise_unfreeze, device, to_compile, weight_decay, learning_rate, out_dir_list)
    
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
            if i >= 2:
                # save some memory by removing unused models
                models[i-2] = None
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
                pseudo_label=pseudo_label)
            models[i] = load_model(model_name_list[i], device, to_compile, out_dir)            
            
if __name__ == '__main__':
    main()
            

            
            

            
            
    
    
