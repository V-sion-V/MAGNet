import torch
import torchvision.transforms as transforms
from torch import no_grad
from torch.utils.data import Dataset, DataLoader
import opt
import utils
from dataset import GSRDataset
from model import GSRNet
import matplotlib.pyplot as plt
from PIL import Image

from opt import batch_size

train_set = GSRDataset(opt.train_dataset_path, opt.SR_factor)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

test_set = GSRDataset(opt.test_dataset_path, opt.SR_factor)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

model = GSRNet(opt.HR_image_size, opt.window_size, opt.num_heads).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(opt.epochs):
    model.train()
    total_train_loss = 0
    for (batch_idx, data) in enumerate(train_loader):
        lr, hr, guide = data["LR"].cuda(), data["HR"].cuda(), data["Guide"].cuda()
        optim.zero_grad()
        pred_hr = model(lr, guide)
        loss = utils.ssim_loss(pred_hr, hr)
        print(loss.item())
        total_train_loss += loss.item()
        loss.backward()
        optim.step()

    model.eval()
    with no_grad:
        total_eval_loss = 0
        for (batch_idx, data) in enumerate(test_loader):
            lr, hr, guide = data["LR"].cuda(), data["HR"].cuda(), data["Guide"].cuda()
            pred_hr = model(lr, guide)
            loss = utils.ssim_loss(pred_hr, hr)
            total_eval_loss += loss.item()
        total_eval_loss /= test_loader.__len__() / batch_size
        print(f"Epoch {epoch} Finished, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}")