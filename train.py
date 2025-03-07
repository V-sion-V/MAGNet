import os
import shutil
import time
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import dataset
import opt
import utils
from dataset import GSRDataset
from model import GSRNet
import matplotlib.pyplot as plt
from PIL import Image

train_set = dataset.get_dataset(train=True)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

eval_set = dataset.get_dataset(train=False)
eval_loader = DataLoader(eval_set, batch_size=opt.batch_size, shuffle=False)

model = GSRNet(opt.HR_image_size, opt.window_size, opt.num_heads).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

start_train_datetime = datetime.now()
start_train_time_str = str(start_train_datetime).split(" ")[0] + '_' + start_train_datetime.strftime("%H-%M-%S")
current_checkpoint_dir = os.path.join(opt.checkpoints_dir, f"GSRNet_{start_train_time_str}")
print(f"Checkpoints saved in directory: {current_checkpoint_dir}")
os.mkdir(current_checkpoint_dir)
shutil.copy("opt.py", os.path.join(current_checkpoint_dir, "opt.txt"))

for epoch in range(1, opt.epochs+1):
    model.train()
    total_train_loss = 0
    range_train_loss = 0
    for (batch_idx, data) in enumerate(train_loader):
        lr, hr, guide = data["LR"].cuda(), data["HR"].cuda(), data["Guide"].cuda()
        optim.zero_grad()
        pred_hr = model(lr, guide)
        loss = utils.calc_loss(pred_hr, hr)
        #print(loss.item())
        total_train_loss += loss.item()
        range_train_loss += loss.item()
        loss.backward()
        optim.step()

        batch_to_print = train_loader.__len__() // opt.print_loss_in_one_epoch
        if batch_idx % batch_to_print == batch_to_print - 1:
            print(f"Epoch: {epoch}, {batch_idx * 1000 // train_loader.__len__() / 10:02.1f}%, "
                  f"Average Train Loss: {range_train_loss / batch_to_print:.16f}")
            range_train_loss = 0

    total_train_loss /= train_loader.__len__()

    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        for (batch_idx, data) in enumerate(eval_loader):
            lr, hr, guide = data["LR"].cuda(), data["HR"].cuda(), data["Guide"].cuda()
            pred_hr = model(lr, guide)
            loss = utils.calc_loss(pred_hr, hr)
            total_eval_loss += loss.item()
        total_eval_loss /= eval_loader.__len__()
        print(f"Epoch {epoch} Finished, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}")

    if epoch % opt.save_model_epoch == opt.save_model_epoch - 1:
        print(f"Epoch {epoch} model saved.")
        torch.save(model.state_dict(), os.path.join(current_checkpoint_dir, f"model{epoch}.pth"))

torch.save(model.state_dict(), os.path.join(current_checkpoint_dir, f"model{opt.epochs}.pth"))