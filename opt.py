import torch

# Data
train_dataset_path = './dataset/train'
eval_dataset_path = './dataset/val'
lr_dir_name = 'thermal/LR_x8'
guide_dir_name = 'visible'
hr_dir_name = 'thermal/GT'
HR_image_size = (448, 640)

# Model
model_name = 'GSRNet'
batch_size = 2
window_size = (7, 10)
num_self_attention_layers = 2
num_cross_attention_layers = 2
num_reconstruction_layers = 2
num_head_list = [6]
num_channels_list = [60]
num_conv_down_layers_list = [2]
num_conv_up_layers_list = [2]
dropout = 0
upsample_mode = 'bicubic' # 'conv_transpose' or 'bicubic'

# Loss
pixel_loss_method = torch.nn.functional.mse_loss
pixel_loss_weight = 1.0
ssim_loss_weight = 0.1
gradient_loss_weight = 0.1

# Train
learning_rate = 0.0004
epochs = 200
print_loss_in_one_epoch = 20
save_model_epoch = 1
checkpoints_dir = 'checkpoints'
progressive = False
start_scale = 1
tensorboard_log_dir = 'tensorboard_log'
lr_decay_step = 24
lr_decay_rate = 0.5

# Device
gpu = torch.device('cuda:0') # Set to cuda:0 in DataParallel

# Current: trained with 2x super resolution