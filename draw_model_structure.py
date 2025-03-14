from main.model import get_model
from main.dataset import get_dataset
import opt
from torchviz import make_dot

model = get_model(opt.model_name)
dataset = get_dataset(mode='train', progressive=opt.progressive, start_scale=opt.start_scale)

lr = dataset[0]["LR"].unsqueeze(0).to(opt.gpu)
guide = dataset[0]["Guide"].unsqueeze(0).to(opt.gpu)
pred = model(lr, guide)

vis = make_dot(pred, params=dict(list(model.named_parameters()) + [('lr', lr), ('guide', guide)]))
vis.format = "png"
vis.directory = "viz"
vis.view()
print(model)