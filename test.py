import torch
from torch.utils.data import DataLoader

import opt
import utils
from dataset import GSRDataset
from model import GSRNet

checkpoint_path = "checkpoints/GSRNet_2025-03-06_21-42-17/model1.pth"

test_set = GSRDataset(opt.test_dataset_path, opt.SR_factor)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

model = GSRNet(opt.HR_image_size, opt.window_size, opt.num_heads).cuda()
model.load_state_dict(torch.load(checkpoint_path))

model.eval()

for (idx, data) in enumerate(test_loader):
    lr, hr, guide = data["LR"].cuda(), data["HR"].cuda(), data["Guide"].cuda()
    pred_hr = model(lr, guide)
    print(utils.ssim_loss(pred_hr, hr).item())