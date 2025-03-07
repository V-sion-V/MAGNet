import os.path

import cv2

import torch
from torch.utils.data import DataLoader

import dataset
import opt
import utils
from dataset import GSRDataset
from model import GSRNet

checkpoint_path = "checkpoints/GSRNet_2025-03-07_12-22-40/model1.pth"
output_path = './output'

test_set = dataset.get_dataset(train=False)
test_loader = DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=False)

model = GSRNet(opt.HR_image_size, opt.window_size, opt.num_heads).cuda()
model.load_state_dict(torch.load(checkpoint_path))

model.eval()

total_ssim = 0
total_psnr = 0

with torch.no_grad():
    for (idx, data) in enumerate(test_loader):
        lr, hr, guide = data["LR"].cuda(), data["HR"].cuda(), data["Guide"].cuda()
        pred_hr = model(lr, guide)
        ssim = utils.ssim(pred_hr, hr).item()
        psnr = utils.psnr(pred_hr, hr).item()
        total_ssim += ssim
        total_psnr += psnr
        print(f"Image {idx} SSIM: {ssim}, PSNR: {psnr}")

        for i in range(opt.test_batch_size):
            pred_hr_img = pred_hr[i].detach().permute(1, 2, 0).cpu().numpy() * 255
            hr_img = hr[i].detach().permute(1, 2, 0).cpu().numpy() * 255

            cv2.imwrite(os.path.join(output_path, "pred", f"{idx}_{i}.png"), pred_hr_img.astype('uint8'))
            cv2.imwrite(os.path.join(output_path, "hr", f"{idx}_{i}.png"), hr_img.astype('uint8'))

    total_ssim /= test_loader.__len__()
    total_psnr /= test_loader.__len__()

    print(f"Average SSIM: {total_ssim}, Average PSNR: {total_psnr}")