import os
import sys
import shutil
from datetime import datetime

import torch
import torchvision.utils
from torch.utils.data import DataLoader

from main import dataset, utils
import opt
from main.model import get_model

from tensorboardX import SummaryWriter
import wandb
import torchvision.utils as vutils
import random

train_set = dataset.get_dataset(mode='train', progressive=opt.progressive, start_scale=opt.start_scale)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

eval_set = dataset.get_dataset(mode='eval', progressive=opt.progressive, start_scale=opt.start_scale)
eval_loader = DataLoader(eval_set, batch_size=opt.batch_size, shuffle=False)

model = get_model(opt.model_name)
if torch.cuda.device_count() > 1 and opt.data_parallel:
    model = torch.nn.DataParallel(model)
model.to(opt.gpu)
optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)

start_train_datetime = datetime.now()
start_train_time_str = str(start_train_datetime).split(" ")[0] + '_' + start_train_datetime.strftime("%H-%M-%S")
current_checkpoint_dir = os.path.join(opt.checkpoints_dir, f"GSRNet_{start_train_time_str}")
print(f"Checkpoints saved in directory: {current_checkpoint_dir}")
os.mkdir(current_checkpoint_dir)
shutil.copy("opt.py", os.path.join(current_checkpoint_dir, "opt.txt"))

if opt.use_tensorboard:
    writer = SummaryWriter(logdir=os.path.join(opt.tensorboard_log_dir, f"GSRNet_{start_train_time_str}"))

if opt.use_wandb:
    wandb.login(key=opt.wandb_key)
    wandb.init(project="MAGNet", name=start_train_time_str, 
               config={"lr_decay_step": opt.lr_decay_step, "lr": opt.learning_rate, "lr_dir_name": opt.lr_dir_name, "window_size": opt.window_size,
                       "num_fusion_layer": opt.num_cross_attention_layers, "num_extract_layer": opt.num_self_attention_layers, "num_reconstruct_layer": opt.num_reconstruction_layers,
                       "multi-scale": len(opt.num_channels_list), "num_channels": opt.num_channels_list, "dataset": opt.train_dataset_path, "data_aug": opt.data_aug}, 
               dir=opt.wandb_log_dir)

for epoch in range(1, opt.epochs+1):
    model.train()
    total_train_loss = 0
    range_train_loss = 0
    for (batch_idx, data) in enumerate(train_loader):
        lr, hr, guide = data["LR"].to(opt.gpu), data["HR"].to(opt.gpu), data["Guide"].to(opt.gpu)
        lr = torch.nn.functional.interpolate(lr, hr.shape[-2:], mode='bicubic', align_corners=False)
        if opt.data_aug:
            shift_h = random.randrange(0, opt.window_size[0])
            shift_w = random.randrange(0, opt.window_size[1])
            lr = lr[:,:,shift_h:lr.shape[2] + shift_h - opt.window_size[0]*8,shift_w:lr.shape[3] + shift_w - opt.window_size[1]*8]
            hr = hr[:,:,shift_h:hr.shape[2] + shift_h-opt.window_size[0]*8,shift_w:hr.shape[3] + shift_w-opt.window_size[1]*8]
            guide = guide[:,:,shift_h:guide.shape[2] + shift_h-opt.window_size[0]*8,shift_w:guide.shape[3] + shift_w-opt.window_size[1]*8]
        optim.zero_grad()
        pred_hr = model(lr, guide)
        pred_hr = torch.clamp(pred_hr, 0, 1)
        loss = utils.calc_loss(pred_hr, hr)
        total_train_loss += loss.detach().item()
        range_train_loss += loss.detach().item()
        loss.backward()
        optim.step()

        if opt.use_tensorboard:
            writer.add_scalar(f'Train/BatchLoss', loss.item(), (epoch-1) * train_loader.__len__() + batch_idx)
        if opt.use_wandb:
            wandb.log({"Train/BatchLoss": loss.item(), "Train/ImageCount": (epoch-1) * train_loader.__len__() + batch_idx})

        batch_to_print = train_loader.__len__() // opt.print_loss_in_one_epoch
        if batch_idx % batch_to_print == batch_to_print - 1:
            print(f"Epoch: {epoch}, {batch_idx * 1000 // train_loader.__len__() / 10:02.1f}%, "
                  f"Average Train Loss: {range_train_loss / batch_to_print:.16f}")
            sys.stdout.flush()
            range_train_loss = 0

    total_train_loss /= train_loader.__len__()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        total_eval_psnr = 0
        total_eval_ssim = 0
        for (batch_idx, data) in enumerate(eval_loader):
            lr, hr, guide = data["LR"].to(opt.gpu), data["HR"].to(opt.gpu), data["Guide"].to(opt.gpu)
            lr = torch.nn.functional.interpolate(lr, hr.shape[-2:], mode='bicubic', align_corners=False)
            pred_hr = model(lr, guide)
            pred_hr = torch.clamp(pred_hr, 0, 1)
            loss = utils.calc_loss(pred_hr, hr)
            total_eval_loss += loss.item()
            total_eval_psnr += utils.psnr(pred_hr, hr).item()
            total_eval_ssim += utils.ssim(pred_hr, hr).item()

            if opt.use_tensorboard:
                for i in range(lr.shape[0]):
                    writer.add_image(f"Eval/Predict{data['Name'][i]}", pred_hr[i], epoch)

        total_eval_loss /= eval_loader.__len__()
        total_eval_psnr /= eval_loader.__len__()
        total_eval_ssim /= eval_loader.__len__()
        print(f"Epoch {epoch} Finished:")
        print(f"Total Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}")
        print(f"Eval PSNR: {total_eval_psnr}, Eval SSIM: {total_eval_ssim}")

        if opt.use_tensorboard:
            writer.add_scalar(f'Train/TotalLoss', total_train_loss, epoch)
            writer.add_scalar(f'Eval/TotalLoss', total_eval_loss, epoch)
            writer.add_scalar(f'Eval/PSNR', total_eval_psnr, epoch)
            writer.add_scalar(f'Eval/SSIM', total_eval_ssim, epoch)
        if opt.use_wandb:
            wandb.log({'Train/TotalLoss':total_train_loss, 'Eval/TotalLoss': total_eval_loss, \
                       'Eval/PSNR': total_eval_psnr, 'Eval/SSIM': total_eval_ssim, 'Epoch': epoch})

    if epoch % opt.save_model_epoch == opt.save_model_epoch - 1:
        print(f"Epoch {epoch} model saved.")
        torch.save(model.state_dict(), os.path.join(current_checkpoint_dir, f"model{epoch}.pth"))

torch.save(model.state_dict(), os.path.join(current_checkpoint_dir, f"model{opt.epochs}.pth"))
if opt.use_wandb:
    wandb.finish()