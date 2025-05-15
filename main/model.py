from asyncio import shield

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

import main.utils as utils
import opt


class WindowSelfAttention(nn.Module):
    def __init__(self, window_size:tuple, num_channels:int, num_heads:int, dropout:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.window_height = window_size[0]
        self.window_width = window_size[1]
        self.num_window_elements = self.window_height * self.window_width
        self.num_channels = num_channels
        self.num_heads = num_heads

        self.w_qkv = nn.Linear(num_channels, num_channels*3)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
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

        self.proj = nn.Linear(num_channels, num_channels)

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(num_channels)

        if num_channels // num_heads != num_channels / num_heads:
            raise ValueError("Number of channels must be divisible by number of heads.")

    def forward(self, x, shift_mask:torch.Tensor=None):
        # B_ = batch size * window count, N = window element count, C = number of channels
        B_, N, C = x.shape
        assert C == self.num_channels, "Number of channels must match."

        # [B_ window element count, total channels]->[qkv, B_, num_heads, window element count, channel in head]
        qkv = self.w_qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # 1, nH, Wh*Ww, Wh*Ww
        
        if shift_mask is not None:
            relative_position_bias = relative_position_bias + shift_mask.repeat(B_//shift_mask.shape[0],1,1).unsqueeze(1)

        # [B_, num_heads, window element count, channel in head] -> [B_, window element count, num_heads, channel in head]
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=relative_position_bias)
        out = out.transpose(1, 2).reshape(B_, N, C)

        out= self.proj(out)

        out = self.dropout(out)

        out = self.norm(out + x)

        return out


class WindowCrossAttention(nn.Module):
    def __init__(self, window_size:tuple, num_channels:int, num_heads:int, dropout:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.window_height = window_size[0]
        self.window_width = window_size[1]
        self.num_window_elements = self.window_height * self.window_width
        self.num_channels = num_channels
        self.num_heads = num_heads

        self.w_qx = nn.Linear(num_channels, num_channels)
        self.w_ky = nn.Linear(num_channels, num_channels)
        self.w_vy = nn.Linear(num_channels, num_channels)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
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

        self.proj = nn.Linear(num_channels, num_channels)

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(num_channels)
        
        if num_channels // num_heads != num_channels / num_heads:
            raise ValueError("Number of channels must be divisible by number of heads.")

    def forward(self, x, y, shift_mask:torch.Tensor=None):
        # B_ = batch size * window count, N = window element count, C = number of channels
        B_, N, C = x.shape
        _B_, _N, _C = y.shape
        assert B_ == _B_ and N == _N and C == _C, "Shape of x and y must match."
        assert C == self.num_channels, "Number of channels must match."

        # [B_ window element count, total channels]->[B_, num_heads, window element count, channel in head]
        qx = self.w_qx(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        ky = self.w_ky(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        vy = self.w_vy(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # 1, nH, Wh*Ww, Wh*Ww
        
        if shift_mask is not None:
            relative_position_bias = relative_position_bias + shift_mask.repeat(B_//shift_mask.shape[0],1,1).unsqueeze(1)

        # [B_, num_heads, window element count, channel in head] -> [B_, window element count, num_heads, channel in head]
        out = nn.functional.scaled_dot_product_attention(qx, ky, vy, attn_mask=relative_position_bias)
        out = out.transpose(1, 2).reshape(B_, N, C)

        out = self.proj(out)

        out = self.dropout(out)

        out = self.norm(out + x)

        return out

class FeedForward(nn.Module):
    def __init__(self, num_channels:int, expansion:int=2, dropout:float=0.0):
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

class AttentionExtractionBlock(nn.Module):
    def __init__(self, window_size:tuple[2], num_channels:int, num_heads:int, expansion:int=2, dropout:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.self_attention1 = WindowSelfAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward2 = FeedForward(num_channels, expansion=expansion, dropout=dropout)

        self.self_attention2 = WindowSelfAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward1 = FeedForward(num_channels, expansion=expansion, dropout=dropout)

    def forward(self, x):
        image_size = x.shape[-2:]
        shift_mask = utils.calculate_shift_mask(image_size, self.window_size)

        x = utils.window_partition(x, self.window_size)
        x = self.self_attention1(x)
        x = self.feed_forward1(x)
        x = utils.window_merge(x, self.window_size, image_size)
        x = utils.half_window_shift(x, self.window_size, 'forward')
        x = utils.window_partition(x, self.window_size)
        
        x = self.self_attention2(x, shift_mask)
        x = self.feed_forward2(x)
        x = utils.window_merge(x, self.window_size, image_size)
        x = utils.half_window_shift(x, self.window_size, 'backward')
        return x

class AttentionFusionBlock(nn.Module):
    def __init__(self, window_size:tuple[2], num_channels:int, num_heads:int, expansion:int=2, dropout:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.self_attention1 = WindowSelfAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward1 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        self.self_attention2 = WindowSelfAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward2 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        self.self_attention3 = WindowSelfAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward3 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        self.self_attention4 = WindowSelfAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward4 = FeedForward(num_channels, expansion=expansion, dropout=dropout)

        self.cross_attention1 = WindowCrossAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward5 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        self.cross_attention2 = WindowCrossAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward6 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        self.cross_attention3 = WindowCrossAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward7 = FeedForward(num_channels, expansion=expansion, dropout=dropout)
        self.cross_attention4 = WindowCrossAttention(window_size, num_channels, num_heads, dropout)
        self.feed_forward8 = FeedForward(num_channels, expansion=expansion, dropout=dropout)

    def forward(self, IN):
        x, y = IN
        image_size = x.shape[-2:]
        shift_mask = utils.calculate_shift_mask(image_size, self.window_size)

        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)

        x = self.self_attention1(x)
        y = self.self_attention2(y)
        x = self.feed_forward1(x)
        y = self.feed_forward2(y)

        x = utils.window_merge(x, self.window_size, image_size)
        y = utils.window_merge(y, self.window_size, image_size)

        x = utils.half_window_shift(x, self.window_size, 'forward')
        y = utils.half_window_shift(y, self.window_size, 'forward')

        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)

        x = self.self_attention3(x, shift_mask)
        y = self.self_attention4(y, shift_mask)
        x = self.feed_forward3(x)
        y = self.feed_forward4(y)

        x = utils.window_merge(x, self.window_size, image_size)
        y = utils.window_merge(y, self.window_size, image_size)

        x = utils.half_window_shift(x, self.window_size, 'backward')
        y = utils.half_window_shift(y, self.window_size, 'backward')

        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)

        new_x = self.cross_attention1(x, y)
        new_y = self.cross_attention2(y, x)
        x = self.feed_forward5(new_x)
        y = self.feed_forward6(new_y)
        
        x = utils.window_merge(x, self.window_size, image_size)
        y = utils.window_merge(y, self.window_size, image_size)

        x = utils.half_window_shift(x, self.window_size, 'forward')
        y = utils.half_window_shift(y, self.window_size, 'forward')

        x = utils.window_partition(x, self.window_size)
        y = utils.window_partition(y, self.window_size)

        new_x = self.cross_attention3(x, y, shift_mask)
        new_y = self.cross_attention4(y, x, shift_mask)
        x = self.feed_forward7(x)
        y = self.feed_forward8(y)

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
    def __init__(self, image_size:tuple[2], window_size:tuple[2], num_head_list:list[int],
                 num_self_attention_layers:int, num_cross_attention_layers:int, num_reconstruction_layers:int,
                 num_channels_list:list[int], num_conv_down_layers_list:list[int], num_conv_up_layers_list:list[int], 
                 dropout:float, upsample_mode:str, num_thermal_channels:int):
        super().__init__()
        self.image_size = image_size
        self.window_size = window_size
        self.num_head_list = num_head_list
        self.num_self_attention_layers = num_self_attention_layers
        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_channels_list = num_channels_list
        self.num_conv_down_layers_list = num_conv_down_layers_list
        self.num_conv_up_layers_list = num_conv_up_layers_list
        self.upsample_mode = upsample_mode
        self.num_unet_layers = len(num_channels_list)
        self.num_reconstruction_layers = num_reconstruction_layers

        assert len(num_channels_list) == len(num_conv_down_layers_list) == len(num_conv_up_layers_list)
        
        self.top_down_conv_ir = ConvBlock(num_thermal_channels, num_channels_list[0], num_conv_down_layers_list[0])
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

        self.self_attention_blocks_ir = nn.ModuleList()
        for i in range(self.num_unet_layers):
            temp = nn.Sequential()
            for j in range(self.num_self_attention_layers):
                temp.append(AttentionExtractionBlock(window_size, num_channels_list[i], num_head_list[i], dropout=dropout))
            self.self_attention_blocks_ir.append(temp)

        self.self_attention_blocks_vi = nn.ModuleList()
        for i in range(self.num_unet_layers):
            temp = nn.Sequential()
            for j in range(self.num_self_attention_layers):
                temp.append(AttentionExtractionBlock(window_size, num_channels_list[i], num_head_list[i], dropout=dropout))
            self.self_attention_blocks_vi.append(temp)

        self.cross_attention_blocks = nn.ModuleList()
        for i in range(self.num_unet_layers):
            temp = nn.Sequential()
            for j in range(self.num_cross_attention_layers):
                temp.append(AttentionFusionBlock(window_size, num_channels_list[i], num_head_list[i], dropout=dropout))
            self.cross_attention_blocks.append(temp)

        self.reconstruction_blocks = nn.ModuleList()
        for i in range(self.num_unet_layers):
            temp = nn.Sequential()
            temp.append(nn.Conv2d(num_channels_list[i] * 2, num_channels_list[i], kernel_size=3, stride=1, padding=1))
            for j in range(self.num_reconstruction_layers):
                temp.append(AttentionExtractionBlock(window_size, num_channels_list[i], num_head_list[i], dropout=dropout))
            self.reconstruction_blocks.append(temp)
        
        self.bottom_up_conv = ConvBlock(num_channels_list[-1], num_channels_list[-1], num_conv_up_layers_list[-1])
        self.up_sample_list = nn.ModuleList()
        for i in range(self.num_unet_layers-2, -1, -1):
            self.up_sample_list.append(UpSampleBlock(upsample_mode, 2, num_channels_list[i+1], num_channels_list[i]))
        self.up_conv_list = nn.ModuleList()
        for i in range(self.num_unet_layers-2, -1, -1):
            self.up_conv_list.append(ConvBlock(2 * num_channels_list[i], num_channels_list[i], num_conv_up_layers_list[i]))
        
        self.proj = nn.Conv2d(num_channels_list[0], num_thermal_channels, kernel_size=1, stride=1, padding=0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized Model With {total_params} Parameters.")


    def forward(self, x:torch.Tensor, y:torch.Tensor):
        original_x = x
        
        # Downward path
        x = self.top_down_conv_ir(x)
        y = self.top_down_conv_vi(y)

        res_list = []

        x = torch.utils.checkpoint.checkpoint(self.self_attention_blocks_ir[0], x, use_reentrant=False)
        y = torch.utils.checkpoint.checkpoint(self.self_attention_blocks_vi[0], y, use_reentrant=False)
        res_x, res_y = torch.utils.checkpoint.checkpoint(self.cross_attention_blocks[0],(x,y),use_reentrant=False)
        res = torch.cat([res_x, res_y], dim=1)
        res = torch.utils.checkpoint.checkpoint(self.reconstruction_blocks[0], res, use_reentrant=False)
        res_list.append(res)

        for i in range(self.num_unet_layers-1):
            x = self.down_max_pool_list_ir[i](x)
            y = self.down_max_pool_list_vi[i](y)
            x = self.down_conv_list_ir[i](x)
            y = self.down_conv_list_vi[i](y)
            x = torch.utils.checkpoint.checkpoint(self.self_attention_blocks_ir[i+1], x ,use_reentrant=False)
            y = torch.utils.checkpoint.checkpoint(self.self_attention_blocks_vi[i+1], y ,use_reentrant=False)
            res_x, res_y = torch.utils.checkpoint.checkpoint(self.cross_attention_blocks[i+1], (x, y), use_reentrant=False)
            res = torch.cat([res_x, res_y], dim=1)
            res = torch.utils.checkpoint.checkpoint(self.reconstruction_blocks[i+1], res, use_reentrant=False)
            res_list.append(res)

        z = res_list[-1]
        
        # Upward path
        z = self.bottom_up_conv(z)
        
        for i in range(self.num_unet_layers-1):
            z = self.up_sample_list[i](z)
            z = torch.cat([z, res_list[-i-2]], dim=1)
            z = self.up_conv_list[i](z)
        
        z = self.proj(z)
        z = torch.nn.functional.tanh(z)
        
        return original_x + z


def get_model(name):
    if name == 'GSRNet':
        return GSRNet(opt.HR_image_size, opt.window_size, opt.num_head_list,
                   opt.num_self_attention_layers, opt.num_cross_attention_layers, opt.num_reconstruction_layers,
                   opt.num_channels_list, opt.num_conv_down_layers_list, opt.num_conv_up_layers_list,
                   opt.dropout, opt.upsample_mode, opt.num_thermal_channels).to(opt.gpu)

