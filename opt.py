# Data
train_dataset_path = './dataset/train'
eval_dataset_path = './dataset/val'
lr_dir_name = 'thermal/LR_x8'
guide_dir_name = 'visible'
hr_dir_name = 'thermal/GT'
HR_image_size = (448, 640)

# Model
batch_size = 2
test_batch_size = 1
window_size = (7, 10)
num_heads = 8
cnn_norm = None

# Loss
l1_loss_weight = 0.8
ssim_loss_weight = 0.1
gradient_loss_weight = 0.1

# Train
epochs = 100
print_loss_in_one_epoch = 20
save_model_epoch = 1
checkpoints_dir = 'checkpoints'

# Eval
output_dir = 'output'
checkpoints_path = "checkpoints/GSRNet_2025-03-07_12-22-40/model1.pth"