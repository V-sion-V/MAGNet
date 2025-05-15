import os.path

import cv2

import torch
from torch.utils.data import DataLoader

from main import dataset, utils
import opt
from main.model import get_model

batch_size = 1
output_dir = "result/output/VGTSR/test"

eval_set = dataset.get_dataset(mode='eval')
eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

def load_model(ckpt_path):
    model = get_model(opt.model_name)

    checkpoint = torch.load(ckpt_path, map_location=opt.gpu)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            new_key = k.replace("module.", "")  # 弱智数据并行
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model

model = load_model("result/checkpoints/GSRNet_2025-05-02_17-32-37/model346.pth")

total_ssim = 0
total_psnr = 0

with torch.no_grad():
    for (idx, data) in enumerate(eval_loader):
        lr, hr, guide = data["LR"].to(opt.gpu), data["HR"].to(opt.gpu), data["Guide"].to(opt.gpu)
        lr = torch.nn.functional.interpolate(lr, hr.shape[-2:], mode='bicubic', align_corners=False)
        pred_hr = model(lr, guide)
        #pred_hr = torch.nn.functional.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        pred_hr = torch.clamp(pred_hr, 0, 1)
        ssim = utils.ssim(pred_hr, hr).item()
        psnr = utils.psnr(pred_hr, hr).item()
        total_ssim += ssim
        total_psnr += psnr
        print(f"{data['Name']}")
        print(f"Image {idx} SSIM: {ssim}, PSNR: {psnr}")

        for i in range(batch_size):
            pred_hr_img = pred_hr[i].detach().permute(1, 2, 0).cpu().numpy() * 255
            pred_hr_img = cv2.cvtColor(pred_hr_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(os.path.join(output_dir, data['Name'][i])), pred_hr_img.astype('uint8'))

    total_ssim /= eval_loader.__len__()
    total_psnr /= eval_loader.__len__()

    print(f"Average SSIM: {total_ssim}, Average PSNR: {total_psnr}")