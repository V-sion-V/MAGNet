from main.model import GSRNet
from main.dataset import get_dataset
import opt
from torchviz import make_dot

model = GSRNet(opt.HR_image_size, opt.window_size, opt.num_heads, opt.num_attention_layers,
                opt.num_channels_list, opt.num_conv_down_layers_list, opt.num_conv_up_layers_list, 
                opt.dropout, opt.upsample_mode).to(opt.gpu)
dataset = get_dataset(mode='train', progressive=opt.progressive, start_scale=opt.start_scale)

lr = dataset[0]["LR"].unsqueeze(0).to(opt.gpu)
guide = dataset[0]["Guide"].unsqueeze(0).to(opt.gpu)
pred = model(lr, guide)

vis = make_dot(pred, params=dict(list(model.named_parameters()) + [('lr', lr), ('guide', guide)]))
vis.format = "png"
vis.directory = "viz"
vis.view()