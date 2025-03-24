import torch

# Data
train_dataset_path = 'dataset/VGTSR/train'
eval_dataset_path = 'dataset/VGTSR/eval'
lr_dir_name = 'LR thermal/8/BI'
guide_dir_name = 'HR RGB'
hr_dir_name = 'GT thermal'
HR_image_size = (512, 640)

# Model
model_name = 'GSRNet'
batch_size = 4
window_size = (8, 10)
num_self_attention_layers = 1
num_cross_attention_layers = 1
num_reconstruction_layers = 1
num_head_list = [4, 8, 16, 32]
num_channels_list = [64, 128, 256, 512]
num_conv_down_layers_list = [2, 2, 2, 2]
num_conv_up_layers_list = [2, 2, 2, 2]
dropout = 0.0
upsample_mode = 'bicubic' # 'conv_transpose' or 'bicubic'
num_thermal_channels = 3

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
data_parallel = True

# Device
gpu = torch.device('cuda:0') # Set to cuda:0 in DataParallel

# Current: trained with 2x super resolution