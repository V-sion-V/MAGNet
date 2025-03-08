import re
import pandas as pd

def find_after_string_in_file(filename, search_string):
    data_list = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):  # 逐行读取
            matchs = re.findall(fr"{search_string}([\d.]+)", line)
            for match in matchs:
                data_list.append(match)
    
    return data_list

# 示例：查找 "hello" 在 "example.txt" 文件中的所有位置
file_path = "../log/train_log_2025-03-08_20-44-41.log"

eval_psnr_list = find_after_string_in_file(file_path, 'Eval PSNR: ')
eval_ssim_list = find_after_string_in_file(file_path, 'Eval SSIM: ')
eval_loss_list = find_after_string_in_file(file_path, 'Eval Loss: ')

df=pd.DataFrame({
    'Eval PSNR': eval_psnr_list,
    'Eval SSIM': eval_ssim_list,
    'Eval Loss': eval_loss_list
})

df.to_csv('eval_metrics.csv', index=False)