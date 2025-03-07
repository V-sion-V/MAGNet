import torch
from sympy import false
from torch import nn
import torch.nn.functional as F

import opt
import utils


class WindowCrossAttention(nn.Module):
    def __init__(self, window_size:tuple, num_channels:int, num_heads:int, dropout:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.window_height = window_size[0]
        self.window_width = window_size[1]
        self.num_window_elements = self.window_height * self.window_width
        self.num_channels = num_channels
        self.num_heads = num_heads

        self.w_qkv1 = nn.Linear(num_channels, num_channels*3)
        self.w_qkv2 = nn.Linear(num_channels, num_channels*3)

        self.relative_position_bias1 = nn.Parameter(
            torch.randn(num_heads, self.num_window_elements,self.num_window_elements))
        self.relative_position_bias2 = nn.Parameter(
            torch.randn(num_heads, self.num_window_elements,self.num_window_elements))

        self.proj1 = nn.Linear(num_channels, num_channels)
        self.proj2 = nn.Linear(num_channels, num_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)

        if num_channels // num_heads != num_channels / num_heads:
            raise ValueError("Number of channels must be divisible by number of heads.")

    def forward(self, x, y):
        # B_ = batch size * window count, N = window element count, C = number of channels
        B_, N, C = x.shape
        _B_, _N, _C = y.shape
        if B_ != _B_ or C != _C or N != _N:
            raise ValueError("Shape of x and y must match.")
        if C != self.num_channels:
            raise ValueError("Number of channels must match.")

        # [B_ window element count, total channels]->[qkv, B_, num_heads, window element count, channel in head]
        qkv1 = self.w_qkv1(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.w_qkv2(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn1 = (q1 @ k2.transpose(-2, -1)) / self.num_channels ** 0.5 # [B_, num_heads, window element count, window element count]
        attn2 = (q2 @ k1.transpose(-2, -1)) / self.num_channels ** 0.5

        attn1 += self.relative_position_bias1.unsqueeze(0)
        attn2 += self.relative_position_bias2.unsqueeze(0)

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        # [B_, num_heads, window element count, channel in head] -> [B_, window element count, num_heads, channel in head]
        out1 = (attn1 @ v2).transpose(1, 2).reshape(B_, N, C)
        out2 = (attn2 @ v1).transpose(1, 2).reshape(B_, N, C)

        out1 = self.proj1(out1)
        out2 = self.proj2(out2)

        out1 = self.dropout1(out1)
        out2 = self.dropout2(out2)

        out1 = self.norm1(out1 + x)
        out2 = self.norm2(out2 + y)

        return out1, out2

class FeedForward(nn.Module):
    def __init__(self, num_channels:int, expansion:int=4, dropout:float=0.0):
        super().__init__()
        self.fc1 = nn.Linear(num_channels, num_channels * expansion)
        self.fc2 = nn.Linear(num_channels * expansion, num_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_channels_in:int, num_channels_hidden:int, num_layers:int, norm_layer:nn.Module=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(num_channels_in, num_channels_hidden, 3, 1, 1))
        if norm_layer is not None:
            self.layers.append(norm_layer)
        for i in range(num_layers-1):
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Conv2d(num_channels_hidden, num_channels_hidden, 3, 1, 1))
            if norm_layer is not None:
                self.layers.append(norm_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GSRNet(nn.Module):
    def __init__(self, image_size:tuple[2], window_size:tuple[2], num_head:int):
        super().__init__()
        self.image_size = image_size
        self.window_size = window_size
        self.num_head = num_head

        self.conv_down_1_ir = ConvBlock(1, 64, 3, opt.cnn_norm)
        self.conv_down_1_vi = ConvBlock(3, 64, 3, opt.cnn_norm)

        self.cross_attention_1 = WindowCrossAttention(window_size, 64, num_head)
        self.feed_forward_1_ir = FeedForward(64)
        self.feed_forward_1_vi = FeedForward(64)
        self.cross_attention_2 = WindowCrossAttention(window_size, 64, num_head)
        self.feed_forward_2_ir = FeedForward(64)
        self.feed_forward_2_vi = FeedForward(64)

        self.conv_up_1 = ConvBlock(128, 128, 3, opt.cnn_norm)
        self.proj = nn.Conv2d(128, 1, 1, 1, 0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized Model With {total_params} Parameters.")

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        x = F.interpolate(x, y.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv_down_1_ir(x)
        y = self.conv_down_1_vi(y)

        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)

        x, y = self.cross_attention_1(x, y)
        x = self.feed_forward_1_ir(x)
        y = self.feed_forward_1_vi(y)

        x = utils.window_merge(x, self.window_size, self.image_size)
        y = utils.window_merge(y, self.window_size, self.image_size)

        x = utils.half_window_shift(x, self.window_size, False)
        y = utils.half_window_shift(y, self.window_size, False)

        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)

        x, y = self.cross_attention_2(x, y)
        x = self.feed_forward_2_ir(x)
        y = self.feed_forward_2_vi(y)

        x = utils.window_merge(x, self.window_size, self.image_size)
        y = utils.window_merge(y, self.window_size, self.image_size)

        x = utils.half_window_shift(x, self.window_size, True)
        y = utils.half_window_shift(y, self.window_size, True)

        z = torch.concat((x, y), dim=1)

        z = self.conv_up_1(z)
        z = self.proj(z)

        return z


