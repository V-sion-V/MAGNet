import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

import main.utils as utils


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

        # define a parameter table of relative position bias
        self.relative_position_bias_table1 = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table2 = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj1 = nn.Linear(num_channels, num_channels)
        self.proj2 = nn.Linear(num_channels, num_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        
        if num_channels // num_heads != num_channels / num_heads:
            raise ValueError("Number of channels must be divisible by number of heads.")

    def forward(self, x, y, mask=None):
        # B_ = batch size * window count, N = window element count, C = number of channels
        B_, N, C = x.shape
        _B_, _N, _C = y.shape
        assert B_ == _B_ and N == _N and C == _C, "Shape of x and y must match."
        assert C == self.num_channels, "Number of channels must match."

        # [B_ window element count, total channels]->[qkv, B_, num_heads, window element count, channel in head]
        qkv1 = self.w_qkv1(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.w_qkv2(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn1 = (q1 @ k2.transpose(-2, -1)) / self.num_channels ** 0.5 # [B_, num_heads, window element count, window element count]
        attn2 = (q2 @ k1.transpose(-2, -1)) / self.num_channels ** 0.5

        relative_position_bias1 = self.relative_position_bias_table1[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias1 = relative_position_bias1.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        relative_position_bias2 = self.relative_position_bias_table2[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias2 = relative_position_bias2.permute(2, 0, 1).contiguous()

        attn1 += relative_position_bias1.unsqueeze(0)
        attn2 += relative_position_bias2.unsqueeze(0)

        if mask is not None: # [num_windows, window element count, window element count]
            nW = mask.shape[0]
            attn1 = attn1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, N, N)
            attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, N, N)

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
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = self.norm(y+x)
        return y

class CrossAttentionBlock(nn.Module):
    def __init__(self, window_size:tuple[2], num_channels:int, num_heads:int, expansion:int=4, dropout:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.cross_attention1 = WindowCrossAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward1 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        
        self.cross_attention2 = WindowCrossAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward2 = FeedForward(num_channels, expansion=expansion, dropout=dropout)

    def forward(self, IN):
        x, y = IN
        image_size = x.shape[-2:]
        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)
        
        x, y = self.cross_attention1(x, y)
        x = self.feed_forward1(x)
        y = self.feed_forward1(y)
        
        x = utils.window_merge(x, self.window_size, image_size)
        y = utils.window_merge(y, self.window_size, image_size)
        
        x = utils.half_window_shift(x, self.window_size, 'forward')
        y = utils.half_window_shift(y, self.window_size, 'forward')
        
        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)
        
        x, y = self.cross_attention2(x, y)
        x = self.feed_forward2(x)
        y = self.feed_forward2(y)
        
        x = utils.window_merge(x, self.window_size, image_size)
        y = utils.window_merge(y, self.window_size, image_size)

        x = utils.half_window_shift(x, self.window_size, 'backward')
        y = utils.half_window_shift(y, self.window_size, 'backward')
        
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, num_channels_in:int, num_channels_hidden:int, num_layers:int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(num_channels_in, num_channels_hidden, 3, 1, 1))
        #self.layers.append(nn.InstanceNorm2d(num_channels_hidden))
        for i in range(num_layers-1):
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Conv2d(num_channels_hidden, num_channels_hidden, 3, 1, 1))
            #self.layers.append(nn.InstanceNorm2d(num_channels_hidden))

    def forward(self, x):
        # [B, C, H, W]
        for layer in self.layers:
            x = layer(x)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, method, scale_factor, num_channels_in:int, num_channels_out:int):
        super().__init__()
        self.method = method
        self.scale_factor = scale_factor
        
        if method == 'bicubic':
            self.conv = nn.Conv2d(num_channels_in, num_channels_out, 1, 1, 0)
        elif method == 'conv_transpose':
            self.convTrans = nn.ConvTranspose2d(num_channels_in, num_channels_out, 2, 2)
        
    def forward(self, x):
        if self.method == 'bicubic':
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
            x = self.conv(x)
        elif self.method == 'conv_transpose':
            x = self.convTrans(x)
        return x
        

class GSRNet(nn.Module):
    def __init__(self, image_size:tuple[2], window_size:tuple[2], num_head:int, num_attention_layers:int,
                 num_channels_list:list[int], num_conv_down_layers_list:list[int], num_conv_up_layers_list:list[int], 
                 dropout:float, upsample_mode:str):
        super().__init__()
        self.image_size = image_size
        self.window_size = window_size
        self.num_head = num_head
        self.num_attention_layers = num_attention_layers
        self.num_channels_list = num_channels_list
        self.num_conv_down_layers_list = num_conv_down_layers_list
        self.num_conv_up_layers_list = num_conv_up_layers_list
        self.upsample_mode = upsample_mode
        self.num_unet_layers = len(num_channels_list)

        assert len(num_channels_list) == len(num_conv_down_layers_list) == len(num_conv_up_layers_list)
        
        self.top_down_conv_ir = ConvBlock(1, num_channels_list[0], num_conv_down_layers_list[0])
        self.down_max_pool_list_ir = nn.ModuleList()
        for i in range(1, self.num_unet_layers):
            self.down_max_pool_list_ir.append(nn.MaxPool2d(2, 2))
        self.down_conv_list_ir = nn.ModuleList()
        for i in range(1, self.num_unet_layers):
            self.down_conv_list_ir.append(ConvBlock(num_channels_list[i-1], num_channels_list[i], num_conv_down_layers_list[i]))

        self.top_down_conv_vi = ConvBlock(3, num_channels_list[0], num_conv_down_layers_list[0])
        self.down_max_pool_list_vi = nn.ModuleList()
        for i in range(1, self.num_unet_layers):
            self.down_max_pool_list_vi.append(nn.MaxPool2d(2, 2))
        self.down_conv_list_vi = nn.ModuleList()
        for i in range(1, self.num_unet_layers):
            self.down_conv_list_vi.append(ConvBlock(num_channels_list[i-1], num_channels_list[i], num_conv_down_layers_list[i]))
        
        self.cross_attention_blocks = nn.ModuleList()
        for i in range(self.num_unet_layers):
            temp = nn.Sequential()
            for j in range(num_attention_layers):
                temp.append(CrossAttentionBlock(window_size, num_channels_list[i], num_head, dropout=dropout))
            self.cross_attention_blocks.append(temp)
        
        self.bottom_up_conv = ConvBlock(num_channels_list[-1], num_channels_list[-1], num_conv_up_layers_list[-1])
        self.up_sample_list = nn.ModuleList()
        for i in range(self.num_unet_layers-2, -1, -1):
            self.up_sample_list.append(UpSampleBlock(upsample_mode, 2, num_channels_list[i+1], num_channels_list[i]))
        self.up_conv_list = nn.ModuleList()
        for i in range(self.num_unet_layers-2, -1, -1):
            self.up_conv_list.append(ConvBlock(2 * num_channels_list[i], num_channels_list[i], num_conv_up_layers_list[i]))
        
        self.proj = nn.Conv2d(num_channels_list[0], 1, kernel_size=1, stride=1, padding=0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized Model With {total_params} Parameters.")


    def forward(self, x:torch.Tensor, y:torch.Tensor):
        x = F.interpolate(x, y.shape[-2:], mode='bicubic', align_corners=False)
        original_x = x
        
        # Downward path
        x = self.top_down_conv_ir(x)
        y = self.top_down_conv_vi(y)
        
        res_list = []
        for i in range(self.num_unet_layers-1):
            res_x, res_y = torch.utils.checkpoint.checkpoint(self.cross_attention_blocks[i],(x,y),use_reentrant=False)
            res_list.append(torch.max(res_x, res_y))
            x = self.down_max_pool_list_ir[i](x)
            y = self.down_max_pool_list_vi[i](y)
            x = self.down_conv_list_ir[i](x)
            y = self.down_conv_list_vi[i](y)
        
        res_x, res_y = self.cross_attention_blocks[-1]((x, y))
        z = torch.max(res_x, res_y)
        
        # Upward path
        z = self.bottom_up_conv(z)
        
        for i in range(self.num_unet_layers-1):
            z = self.up_sample_list[i](z)
            z = torch.cat([z, res_list[-i-1]], dim=1)
            z = self.up_conv_list[i](z)
        
        z = self.proj(z)
        z = torch.nn.functional.tanh(z)
        
        return original_x + z


