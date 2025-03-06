import os
import shutil
import cv2

input_path_train = '../data/VEDAI512/train'
input_path_test = '../data/VEDAI512/test'
output_path = '../dataset'

if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.makedirs(output_path)

os.mkdir(os.path.join(output_path, 'train'))
os.mkdir(os.path.join(output_path, 'test'))

os.mkdir(os.path.join(output_path, 'train', 'HR'))
os.mkdir(os.path.join(output_path, 'train', 'LR2'))
os.mkdir(os.path.join(output_path, 'train', 'LR4'))
os.mkdir(os.path.join(output_path, 'train', 'LR8'))
os.mkdir(os.path.join(output_path, 'train', 'Guide'))

os.mkdir(os.path.join(output_path, 'test', 'HR'))
os.mkdir(os.path.join(output_path, 'test', 'LR2'))
os.mkdir(os.path.join(output_path, 'test', 'LR4'))
os.mkdir(os.path.join(output_path, 'test', 'LR8'))
os.mkdir(os.path.join(output_path, 'test', 'Guide'))

def split_to_256(img):
    img_split_list = []
    for i in range(0, img.shape[0], 256):
        for j in range(0, img.shape[1], 256):
            img_split_list.append(img[i:i+256, j:j+256])
    return img_split_list

def process_file(dir_name, file_name, dest_dir):
    name, type = file_name.split('_')
    if type == 'co.png':
        hr_vi = cv2.imread(os.path.join(dir_name, file_name))
        hr_vi_split_list = split_to_256(hr_vi)
        for i, hr_vi_split in enumerate(hr_vi_split_list):
            cv2.imwrite(os.path.join(dest_dir, 'Guide', name + f'_{i}.png'), hr_vi_split)
    elif type == 'ir.png':
        hr_ir = cv2.imread(os.path.join(dir_name, file_name))
        hr_ir_split_list = split_to_256(hr_ir)
        for i, hr_ir_split in enumerate(hr_ir_split_list):
            cv2.imwrite(os.path.join(dest_dir, 'HR', name + f'_{i}.png'), hr_ir_split)
            for j in [2,4,8]:
                lr = cv2.resize(hr_ir_split, (hr_ir_split.shape[1]//j, hr_ir_split.shape[0]//j))
                cv2.imwrite(os.path.join(dest_dir, f'LR{j}', name + f'_{i}.png'), lr)

for file_name in os.listdir(input_path_train):
    process_file(input_path_train, file_name, os.path.join(output_path, 'train'))

for file_name in os.listdir(input_path_test):
    process_file(input_path_test, file_name, os.path.join(output_path, 'test'))
