import torch
import torchmetrics.image
import torchvision
from torch import nn

import opt


saved_mask = {(0, 0) : None}

def window_partition (x:torch.Tensor, window_size:tuple): #[B, C, H, W] -> [B_, N, C]
    B, C, H, W = x.shape
    window_height, window_width = window_size
    assert H % window_height == 0 and W % window_width == 0, "Image height and width must be divisible by window height and width."
    x = x.view(B, C, H // window_height, window_height, W // window_width, window_width)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_height * window_width, C)
    return x

def window_merge (x:torch.Tensor, window_size:tuple, image_size:tuple): #[B_, N, C] -> [B, C, H, W]
    B_, N, C = x.shape
    window_height, window_width = window_size
    image_height, image_width = image_size
    x = x.view(-1, image_height // window_height, image_width // window_width, window_height, window_width, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(-1, C, image_height, image_width)
    return x

def half_window_shift (x:torch.Tensor, window_size:tuple, direction:str): #[B, C, H, W] -> [B, C, H, W]
    B, C, H, W = x.shape
    window_height, window_width = window_size
    assert H % window_height == 0 and W % window_width == 0, "Image height and width must be divisible by window height and width."
    if direction == 'forward':
        return torch.roll(x, shifts=(-window_height//2, -window_width//2), dims=(2, 3))
    elif direction == 'backward':
        return torch.roll(x, shifts=(window_height//2, window_width//2), dims=(2, 3))
    else:
        raise ValueError('Direction must be either "forward" or "backward".')

def calculate_shift_mask(image_size:tuple, window_size:tuple):
    # calculate attention mask for SW-MSA
    if saved_mask.get(image_size) is not None:
        return saved_mask[image_size]

    H, W = image_size
    window_height, window_width = window_size
    shift_height = window_height // 2
    shift_width = window_width // 2
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -window_height),
                slice(-window_height, -shift_height),
                slice(-shift_height, None))
    w_slices = (slice(0, -window_width),
                slice(-window_width, -shift_width),
                slice(-shift_width, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask.permute(0, 3, 1, 2), window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_height * window_width)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0)).to(opt.gpu)

    saved_mask[image_size] = attn_mask

    return attn_mask

ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(opt.gpu)
def ssim_loss (x:torch.Tensor, y:torch.Tensor):
    return 1 - ssim(x, y)

def gradient_loss (x:torch.Tensor, y:torch.Tensor):
    x_grad_x, x_grad_y = torch.gradient(x, dim=(-2, -1))
    y_grad_x, y_grad_y = torch.gradient(y, dim=(-2, -1))

    loss_x = torch.nn.functional.l1_loss(x_grad_x, y_grad_x)
    loss_y = torch.nn.functional.l1_loss(x_grad_y, y_grad_y)
    return loss_x + loss_y

def calc_loss (pred:torch.Tensor, target:torch.Tensor):
    loss = torch.zeros(1, device=opt.gpu)
    if opt.pixel_loss_weight > 0:
        loss += opt.pixel_loss_weight * opt.pixel_loss_method(pred, target)
    if opt.ssim_loss_weight > 0:
        loss += opt.ssim_loss_weight * ssim_loss(pred, target)
    if opt.gradient_loss_weight > 0:
        loss += opt.gradient_loss_weight * gradient_loss(pred, target)
    return loss

def psnr(pred:torch.Tensor, target:torch.Tensor, max_val=1.0):
    mse = nn.functional.mse_loss(pred, target, reduction='mean')
    psnr_value = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_value
