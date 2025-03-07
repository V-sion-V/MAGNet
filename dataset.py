import os.path
import torch
import torch.utils.data as data
import cv2
import torchvision.transforms
from numpy import dtype

import opt


class GSRDataset(data.Dataset):
    def __init__(self, lr_dir, guide_dir, hr_dir):
        super(GSRDataset, self).__init__()

        self.hr_ir_path = hr_dir
        self.hr_rgb_path = guide_dir
        self.lr_ir_path = lr_dir

        self.file_name_list = os.listdir(self.hr_ir_path)

    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()

        hr_ir = cv2.imread(os.path.join(self.hr_ir_path, self.file_name_list[index]), cv2.IMREAD_GRAYSCALE)
        hr_ir = transform(hr_ir)

        hr_rgb = cv2.imread(os.path.join(self.hr_rgb_path, self.file_name_list[index]))
        hr_rgb = cv2.cvtColor(hr_rgb, cv2.COLOR_BGR2RGB)
        hr_rgb = transform(hr_rgb)

        lr_ir = cv2.imread(os.path.join(self.lr_ir_path, self.file_name_list[index]), cv2.IMREAD_GRAYSCALE)
        lr_ir = transform(lr_ir)

        return {"Name": self.file_name_list[index], "LR":lr_ir, "Guide":hr_rgb, "HR":hr_ir}

    def __len__(self):
        return self.file_name_list.__len__()


def get_dataset(train:bool):
    if train:
        return GSRDataset(os.path.join(opt.train_dataset_path, opt.lr_dir_name),
                       os.path.join(opt.train_dataset_path, opt.guide_dir_name),
                       os.path.join(opt.train_dataset_path, opt.hr_dir_name))
    else:
        return GSRDataset(os.path.join(opt.test_dataset_path, opt.lr_dir_name),
                       os.path.join(opt.test_dataset_path, opt.guide_dir_name),
                       os.path.join(opt.test_dataset_path, opt.hr_dir_name))