import os.path

import cv2

file_name_list = ["00240i.png", "00241i.png", "00242i.png", "00243i.png", "00244i.png",
                  "00245i.png", "00246i.png", "00247i.png", "00248i.png", "00249i.png",]
in_dir = '../dataset/VGTSR/train/GT thermal'
out_dir = '../dataset/VGTSR/train/LR thermal/4/BI'

def downsample(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

for file_name in file_name_list:
    file_path = os.path.join(in_dir, file_name)
    if os.path.exists(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = downsample(img, 0.25)
        out_path = os.path.join(out_dir, file_name.split('.')[0]+'x4.png')
        cv2.imwrite(out_path, img)


