import torch

# Data
train_dataset_path = './dataset/train'
eval_dataset_path = './dataset/val'
lr_dir_name = 'thermal/LR_x8'
guide_dir_name = 'visible'
hr_dir_name = 'thermal/GT'
HR_image_size = (448, 640)

# Model
batch_size = 2
window_size = (7, 10)
num_heads = 8
num_attention_layers = 2
num_channels_list = [64, 128, 256, 512]
num_conv_down_layers_list = [2, 2, 2, 2]
num_conv_up_layers_list = [2, 2, 2, 2]
dropout = 0.5
upsample_mode = 'bicubic' # 'conv_transpose' or 'bicubic'

# Loss
pixel_loss_method = torch.nn.functional.mse_loss
pixel_loss_weight = 1.0
ssim_loss_weight = 0.1
gradient_loss_weight = 0.1

# Train
learning_rate = 0.0001
epochs = 200
print_loss_in_one_epoch = 20
save_model_epoch = 1
checkpoints_dir = 'checkpoints'
progressive = False
start_scale = 1
tensorboard_log_dir = 'tensorboard_log'
lr_decay_step = 16
lr_decay_rate = 0.5

# Device
gpu = torch.device('cuda:0') # Set to cuda:0 in DataParallel

# Current: trained with 2x super resolution