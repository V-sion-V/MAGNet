# Data
train_dataset_path = './dataset/train'
test_dataset_path = './dataset/test'
lr_dir_name = 'LR4'
guide_dir_name = 'Guide'
hr_dir_name = 'HR'
HR_image_size = (256, 256)

# Model
batch_size = 2
test_batch_size = 1
window_size = (8, 8)
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