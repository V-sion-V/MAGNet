import torch

# Data
train_dataset_path = 'dataset/VGTSR/train'
eval_dataset_path = 'dataset/VGTSR/eval'
lr_dir_name = 'LR thermal/4/BD'
guide_dir_name = 'HR RGB'
hr_dir_name = 'GT thermal'
HR_image_size = (512, 640)

# Model
model_name = 'GSRNet'
batch_size = 1
window_size = (8, 8)
num_self_attention_layers = 2
num_cross_attention_layers = 1
num_reconstruction_layers = 4
num_head_list = [4, 8, 16]
num_channels_list = [64, 128, 256]
num_conv_down_layers_list = [2, 2, 2]
num_conv_up_layers_list = [2, 2, 2]
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
epochs = 500
print_loss_in_one_epoch = 20
save_model_epoch = 1
checkpoints_dir = 'result/checkpoints'
progressive = False
start_scale = 1
data_aug =  True


lr_decay_step = 50
lr_decay_rate = 0.5
data_parallel = False

# Display
tensorboard_log_dir = 'result/tensorboard_log'
use_tensorboard = False
wandb_log_dir = 'result/wandb'
use_wandb = True
wandb_key = '5335b0bee32e894ea005755958de07f444f0c459'

# Device
gpu = torch.device('cuda:1') # Set to cuda:0 in DataParallel