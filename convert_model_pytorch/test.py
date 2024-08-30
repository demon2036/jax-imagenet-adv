from typing import Tuple

import timm.models.convnext
from timm.layers import create_conv2d
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
import einops

from .check_conv import convert_flax_to_torch_conv,convert_torch_to_flax_conv

from .check_convnext_stage import convert_flax_to_torch_conv_next_stage,convert_torch_to_flax_conv_next_stage
from .check_layernorm import convert_flax_to_torch_layer_norm,convert_torch_to_flax_layer_norm
from .check_mlp import convert_flax_to_torch_mlp, convert_flax_to_torch_fc,convert_torch_to_flax_fc
from models import ConvNeXt

jax.config.update('jax_platform_name', 'cpu')





def convert_flax_to_torch_conv_next(flax_params, prefix='', sep=''):
    stem_conv = convert_flax_to_torch_conv(flax_params['stem']['layers_0'], prefix='stem.0')
    stem_layer_norm = convert_flax_to_torch_layer_norm(flax_params['stem']['layers_1'], prefix='stem.1')

    state_dict = {**stem_conv, **stem_layer_norm}

    i = 0
    while f'stages_{i}' in flax_params:
        state_dict = state_dict | convert_flax_to_torch_conv_next_stage(flax_params[f'stages_{i}'],
                                                                        prefix=f'stages.{i}', sep='.')
        i += 1

    head_layer_norm = convert_flax_to_torch_layer_norm(flax_params['norm'], prefix='head.norm')
    head_fc = convert_flax_to_torch_fc(flax_params['head'], prefix='head.fc')

    state_dict = state_dict | {**head_fc, **head_layer_norm}
    print(state_dict.keys())
    # while True:
    #     pass
    state_dict = {f'{prefix}{sep}{k}': v for k, v in state_dict.items()}
    return state_dict


def test():
    b, h, w, c = shape = 1, 224, 224, 3

    rng = jax.random.PRNGKey(0)
    # x = jnp.ones(shape)
    x = np.asarray(jax.random.normal(rng, shape))
    #

    out_c = c
    kernel_size = 7
    padding = 3
    stride = 1
    use_bias = True

    # conv_jax = nn.Conv(out_c, (kernel_size, kernel_size),
    #                    use_bias=use_bias,
    #                    feature_group_count=c,
    #                    # precision='highest',
    #                    padding=[(padding, padding), (padding, padding)], strides=(stride, stride))

    conv_jax = ConvNeXt()

    conv_jax_params = conv_jax.init(rng, x)['params']
    # print(flax.traverse_util.flatten_dict(conv_jax_params, sep='.').keys())
    print(conv_jax_params.keys())

    # model_torch = ConvNeXtBlock(c, out_c)

    out_jax = conv_jax.apply({'params': conv_jax_params}, x,)
    out_jax_np = np.array(out_jax)

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = timm.models.convnext.ConvNeXt(ls_init_value=1)
    print(model_torch.state_dict().keys())



    state_dict = convert_flax_to_torch_conv_next(conv_jax_params, sep='')

    model_torch.load_state_dict(state_dict)

    # while True:
    #     padding

    # print(conv_torch.weight.shape, conv_torch.state_dict())
    #
    # while True:
    #     padding
    out_torch = model_torch(x_torch)

    out_torch_np = out_torch.detach().numpy()
    # out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    print(np.argmax(out_torch_np), np.argmax(out_jax_np))

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)







def convert_torch_to_flax_conv_next(torch_params, prefix='', sep=''):
    stages = torch_params.pop('stages')
    for k, v in stages.items():
        torch_params[f'stages_{k}'] = v

    print(torch_params.keys(),torch_params['head']['fc'].keys())


    stem_conv = convert_torch_to_flax_conv(torch_params['stem']['0'], prefix='',sep='')
    stem_layer_norm = convert_torch_to_flax_layer_norm(torch_params['stem']['1'], prefix='',sep='')

    stem={'layers_0':stem_conv,'layers_1':stem_layer_norm}


    state_dict = {'stem':stem}





    i = 0
    while f'stages_{i}' in torch_params:
        # state_dict = state_dict | convert_flax_to_torch_conv_next_stage(torch_params[f'stages_{i}'],
        #                                                                 prefix=f'stages.{i}', sep='.')
        state_dict[f'stages_{i}']=convert_torch_to_flax_conv_next_stage(torch_params[f'stages_{i}'],prefix='',sep='')
        i += 1


    state_dict['norm'] = convert_torch_to_flax_layer_norm(torch_params['head']['norm'], prefix='',sep='')
    state_dict['head'] = convert_torch_to_flax_fc(torch_params['head']['fc'], prefix='',sep='')


    state_dict = {f'{prefix}{sep}{k}': v for k, v in state_dict.items()}
    return state_dict


def test_torch():
    b, h, w, c = shape = 1, 224,224,3

    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)

    out_c = c * 2
    kernel_size = 7
    padding = 3
    stride = 2
    use_bias = True

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = timm.models.convnext.ConvNeXt(c,ls_init_value=1 )
    print(model_torch.state_dict().keys())

    out_torch = model_torch(x_torch)

    # x = np.asarray(jax.random.normal(rng, shape))

    model_jax = ConvNeXt(  ls_init_value=1)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    model_jax_params = convert_torch_to_flax_conv_next(params, sep='')
    # print(params)

    out_jax = model_jax.apply({'params': model_jax_params}, x)
    out_jax_np = np.array(out_jax)

    out_torch_np = out_torch.detach().numpy()
    # out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)


if __name__ == "__main__":
    test()
