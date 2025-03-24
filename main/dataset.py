import os.path
import torch
import torch.utils.data as data
import cv2
import torchvision.transforms
from numpy import dtype

import opt
from opt import lr_dir_name, guide_dir_name, eval_dataset_path, train_dataset_path, hr_dir_name


class GSRDataset(data.Dataset):
    def __init__(self, lr_dir, guide_dir, hr_dir, progressive=False, start_scale = 1, thermal_color_mode='Gray'):
        super(GSRDataset, self).__init__()

        self.hr_ir_path = hr_dir
        self.hr_rgb_path = guide_dir
        self.lr_ir_path = lr_dir
        
        self.progressive = progressive
        self.start_scale = start_scale

        self.thermal_color_mode = thermal_color_mode

        self.hr_ir_file_name_list = sorted(os.listdir(self.hr_ir_path))
        self.hr_rgb_file_name_list = sorted(os.listdir(self.hr_rgb_path))
        self.lr_ir_file_name_list = sorted(os.listdir(self.lr_ir_path))

        assert(self.hr_ir_file_name_list.__len__() == self.hr_rgb_file_name_list.__len__())
        assert(self.hr_ir_file_name_list.__len__() == self.lr_ir_file_name_list.__len__())
        
        hr_ir = cv2.imread(os.path.join(self.hr_ir_path, self.hr_ir_file_name_list[0]), cv2.IMREAD_GRAYSCALE)
        lr_ir = cv2.imread(os.path.join(self.lr_ir_path, self.lr_ir_file_name_list[0]), cv2.IMREAD_GRAYSCALE)
        self.hr_lr_ratio = hr_ir.shape[0] // lr_ir.shape[0]
        
        assert(hr_ir.shape[0] == lr_ir.shape[0] * self.hr_lr_ratio)
        assert(hr_ir.shape[1] == lr_ir.shape[1] * self.hr_lr_ratio)
        
    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()

        if self.thermal_color_mode == 'Gray':
            hr_ir = cv2.imread(os.path.join(self.hr_ir_path, self.hr_ir_file_name_list[index]), cv2.IMREAD_GRAYSCALE)
        else:
            hr_ir = cv2.imread(os.path.join(self.hr_ir_path, self.hr_ir_file_name_list[index]), cv2.IMREAD_COLOR)
            hr_ir = cv2.cvtColor(hr_ir, cv2.COLOR_BGR2RGB)
        hr_ir = transform(hr_ir)
        if self.progressive and self.start_scale < self.hr_lr_ratio//2:
            hr_ir = torch.nn.functional.interpolate(hr_ir.unsqueeze(0), scale_factor=2 * self.start_scale/self.hr_lr_ratio, mode='bicubic', align_corners=False).squeeze(0)

        hr_rgb = cv2.imread(os.path.join(self.hr_rgb_path, self.hr_rgb_file_name_list[index]))
        hr_rgb = cv2.cvtColor(hr_rgb, cv2.COLOR_BGR2RGB)
        hr_rgb = transform(hr_rgb)

        if not self.progressive or self.start_scale == 1:
            if self.thermal_color_mode == 'Gray':
                lr_ir = cv2.imread(os.path.join(self.lr_ir_path, self.lr_ir_file_name_list[index]), cv2.IMREAD_GRAYSCALE)
            else:
                lr_ir = cv2.imread(os.path.join(self.lr_ir_path, self.lr_ir_file_name_list[index]), cv2.IMREAD_COLOR)
                lr_ir = cv2.cvtColor(lr_ir, cv2.COLOR_BGR2RGB)
            lr_ir = transform(lr_ir)
        else:
            lr_ir = torch.nn.functional.interpolate(hr_ir.unsqueeze(0), scale_factor=0.5, mode='bicubic', align_corners=False).squeeze(0)

        return {"Name": self.lr_ir_file_name_list[index], "LR":lr_ir, "Guide":hr_rgb, "HR":hr_ir}

    def __len__(self):
        return self.lr_ir_file_name_list.__len__()


def get_dataset(mode:str, progressive = False, start_scale = 1):
    if mode == 'train':
        return GSRDataset(os.path.join(train_dataset_path, lr_dir_name),
                          os.path.join(train_dataset_path, guide_dir_name),
                          os.path.join(train_dataset_path, hr_dir_name),
                          progressive=progressive,
                          start_scale=start_scale,
                          thermal_color_mode = 'Gray' if opt.num_thermal_channels == 1 else 'RGB')
    elif mode == 'eval':
        return GSRDataset(os.path.join(eval_dataset_path, lr_dir_name),
                          os.path.join(eval_dataset_path, guide_dir_name),
                          os.path.join(eval_dataset_path, hr_dir_name),
                          progressive=progressive,
                          start_scale=start_scale,
                          thermal_color_mode = 'Gray' if opt.num_thermal_channels == 1 else 'RGB')