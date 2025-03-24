import os


def move_files(input_dir, output_dir, amount):
    file_names = sorted(os.listdir(input_dir))
    step = len(file_names)/amount
    count = 0
    for (idx, file_name) in enumerate(file_names):
        file_path = os.path.join(input_dir, file_name)
        if idx > count * step:
            os.rename(file_path, os.path.join(output_dir, file_name))
            count+=1


move_files("../dataset/VGTSR/GT thermal", "../dataset/VGTSR/eval/GT thermal", 225)
move_files("../dataset/VGTSR/HR RGB", "../dataset/VGTSR/eval/HR RGB", 225)
move_files("../dataset/VGTSR/LR thermal/4/BI", "../dataset/VGTSR/eval/LR thermal/4/BI", 225)
move_files("../dataset/VGTSR/LR thermal/4/BD", "../dataset/VGTSR/eval/LR thermal/4/BD", 225)
move_files("../dataset/VGTSR/LR thermal/8/BI", "../dataset/VGTSR/eval/LR thermal/8/BI", 225)
move_files("../dataset/VGTSR/LR thermal/8/BD", "../dataset/VGTSR/eval/LR thermal/8/BD", 225)