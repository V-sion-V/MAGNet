import torch
import torchmetrics.image
import torchvision
from torch import nn

import opt


def window_partition (x:torch.Tensor, window_size:tuple): #[B, C, H, W] -> [B_, N, C]
    B, C, H, W = x.shape
    window_height, window_width = window_size
    if H % window_height != 0 or W % window_width != 0:
        raise ValueError("Image height and width must be divisible by window height and width.")
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

def half_window_shift (x:torch.Tensor, window_size:tuple, reverse:bool): #[B, C, H, W] -> [B, C, H, W]
    B, C, H, W = x.shape
    window_height, window_width = window_size
    if H % window_height != 0 or W % window_width != 0:
        raise ValueError("Image height and width must be divisible by window height and width.")
    if reverse:
        return torch.roll(x, shifts=(-window_height//2, -window_width//2), dims=(2, 3))
    else:
        return torch.roll(x, shifts=(window_height//2, window_width//2), dims=(2, 3))

ssim = torchmetrics.image.StructuralSimilarityIndexMeasure().cuda()
def ssim_loss (x:torch.Tensor, y:torch.Tensor):
    return 1 - ssim(x, y)

def gradient_loss (x:torch.Tensor, y:torch.Tensor):
    x_grad_x, x_grad_y = torch.gradient(x, dim=(-2, -1))
    y_grad_x, y_grad_y = torch.gradient(y, dim=(-2, -1))

    loss_x = torch.nn.functional.l1_loss(x_grad_x, y_grad_x)
    loss_y = torch.nn.functional.l1_loss(x_grad_y, y_grad_y)
    return loss_x + loss_y

def calc_loss (pred:torch.Tensor, target:torch.Tensor):
    return (opt.l1_loss_weight * nn.functional.l1_loss(pred, target) +
            opt.ssim_loss_weight * ssim_loss(pred, target) +
            opt.gradient_loss_weight * gradient_loss(pred, target))

def psnr(pred:torch.Tensor, target:torch.Tensor, max_val=1.0):
    mse = nn.functional.mse_loss(pred, target, reduction='mean')
    psnr_value = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_value