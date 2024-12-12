import logging
import warnings

import torch
from timm.models import VisionTransformer

from timm.models.metaformer import MetaFormer,SepConv
from timm.models.convnext import convnext_xxlarge,convnext_xlarge
from robustbench import load_model
from ptflops import get_model_complexity_info
import torchprofile
# warnings.filterwarnings("ignore")

input_tensor = torch.randn(1, 3, 224, 224)



dims= [128, 256, 512, 768]

for i,dim in enumerate(dims):
    dims[i]=int(dims[i]*2)


# net=MetaFormer(token_mixers=SepConv,depths=(3,12,18,3),dims=(128,256,512,768))
net=MetaFormer(token_mixers=SepConv,depths=(3,12,18,3),dims=dims)
# net=convnext_xxlarge()

net.eval()

# flops = torchprofile.profile_macs(net, input_tensor)
# print("FLOPs:", flops)


flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True,backend='aten',)
print("FLOPs:", flops)
print("Params:", params)

# macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, backend='pytorch',
#                                          print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, backend='aten',
#                                          print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
