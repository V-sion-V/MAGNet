import os.path
import torch
import torch.utils.data as data
import cv2
import torchvision.transforms
from numpy import dtype


class GSRDataset(data.Dataset):
    def __init__(self, path:str, sr_factor:int):
        super(GSRDataset, self).__init__()
        self.path = path

        self.hr_ir_path = os.path.join(path, 'HR')
        self.hr_rgb_path = os.path.join(path, 'Guide')
        self.lr_ir_path = os.path.join(path, 'LR'+str(sr_factor))

        self.file_name_list = os.listdir(self.hr_ir_path)

    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()

        hr_ir = cv2.imread(os.path.join(self.hr_ir_path, self.file_name_list[index]))
        hr_ir = cv2.cvtColor(hr_ir, cv2.COLOR_BGR2RGB)
        hr_ir = transform(hr_ir)

        hr_rgb = cv2.imread(os.path.join(self.hr_rgb_path, self.file_name_list[index]))
        hr_rgb = cv2.cvtColor(hr_rgb, cv2.COLOR_BGR2RGB)
        hr_rgb = transform(hr_rgb)

        lr_ir = cv2.imread(os.path.join(self.lr_ir_path, self.file_name_list[index]))
        lr_ir = cv2.cvtColor(lr_ir, cv2.COLOR_BGR2RGB)
        lr_ir = transform(lr_ir)

        return {"LR":lr_ir, "Guide":hr_rgb, "HR":hr_ir}

    def __len__(self):
        return self.file_name_list.__len__()