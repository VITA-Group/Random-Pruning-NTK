from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.registry
from models.mnist_lenet import Model
from change_width import almost_even_split, replicate_mask_state_dict
from models.initializers import kaiming_normal

lenet_150_50 = Model([150, 50], kaiming_normal)
lenet_300_100 = Model([300,100], kaiming_normal)

parent_dir = "/u/hy6385/open_lth_data/lottery_a050d4cded0448e379626e47a9e737e0" # lenet_150_50
child_dir = "/u/hy6385/open_lth_data/lottery_108ebe7f8e2dbe540dbed0e9011edec1" # lenet_300_100
parent_model = torch.load(parent_dir + "/replicate_1/level_10/main/model_ep0_it0.pth")
child_model = torch.load(child_dir + "/replicate_1/level_0/main/model_ep0_it0.pth")
source_mask = torch.load(parent_dir + "/replicate_1/level_10/main/mask.pth")
target_mask = torch.load(child_dir + "/replicate_1/level_0/main/mask.pth")

delta_scale = 0
lenet300_dict = almost_even_split(parent_model, child_model, delta_scale=delta_scale)
lenet300_mask = replicate_mask_state_dict(source_mask, target_mask)
lenet_150_50.load_state_dict(parent_model)
lenet_300_100.load_state_dict(lenet300_dict)

torch.manual_seed(101)
input = torch.randn(1,28*28)
lenet_150_50.eval()
lenet_300_100.eval()
output_lenet150 = lenet_150_50(input)
output_lenet300 = lenet_300_100(input)
print(torch.norm(output_lenet150 - output_lenet300))