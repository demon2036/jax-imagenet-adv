import functools
import logging
import warnings

import torch
from timm.models import VisionTransformer
from timm.models.vision_transformer import  vit_giant_patch14_224,vit_huge_patch14_clip_336
from timm.models.metaformer import MetaFormer, SepConv, Attention, LayerNorm2dNoBias, LayerNormNoBias
from timm.models.convnext import convnext_xxlarge,convnext_xlarge
# from robustbench import load_model
from ptflops import get_model_complexity_info
# import torchprofile
# warnings.filterwarnings("ignore")

input_tensor = torch.randn(1, 3, 224, 224)

# Attention=functools.partial(Attention,head_dim=128)

dims= [128, 256, 512, 768]

for i,dim in enumerate(dims):
    dims[i]=int(dims[i]*2)
print(dims)

# net=MetaFormer(token_mixers=SepConv,depths=(3,12,18,3),dims=(128,256,512,768))
net=MetaFormer(
    token_mixers=[SepConv, SepConv, Attention, Attention],
    # token_mixers=[SepConv, SepConv, SepConv, SepConv],
    norm_layers=[LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
    # token_mixers=[SepConv,SepConv,Attention,Attention],
    # depths=(3, 12, 18, 3),
               depths=(8,8,24,8),
    dims=dims)
# net=convnext_xxlarge()
# net=vit_giant_patch14_224()
# net=vit_huge_patch14_clip_336()
net.eval()

# flops = torchprofile.profile_macs(net, input_tensor)
# print("FLOPs:", flops)

input_res=(3, 224, 224)
input_res=(3, 336, 336)
flops, params = get_model_complexity_info(net, input_res, as_strings=True, print_per_layer_stat=True,backend='aten',)
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
