import timm.models.convnext
from timm.layers import create_conv2d, LayerNorm2d
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
import einops

from models.convnext import ConvNeXtStage
from .check_conv import convert_flax_to_torch_conv,convert_torch_to_flax_conv

from .check_convnext_block import  convert_flax_to_torch_conv_next_block,convert_torch_to_flax_conv_next_block
from .check_layernorm import convert_flax_to_torch_layer_norm,convert_torch_to_flax_layer_norm

# jax.config.update('jax_platform_name', 'cpu')









def convert_flax_to_torch_conv_next_stage(flax_params, prefix='', sep='.'):
    state_dict = {}
    if 'downsample' in flax_params:
        down_sample_layer_norm = convert_flax_to_torch_layer_norm(flax_params['downsample']['layers_0'],
                                                                  prefix='downsample.0')
        down_sample_conv = convert_flax_to_torch_conv(flax_params['downsample']['layers_1'], prefix='downsample.1')

        state_dict = state_dict | {**down_sample_conv, **down_sample_layer_norm}

    i = 0
    while f'blocks_{i}' in flax_params:
        state_dict = state_dict | convert_flax_to_torch_conv_next_block(flax_params[f'blocks_{i}'],
                                                                        prefix=f'blocks.{i}', sep='.')

        i += 1
    state_dict = {f'{prefix}{sep}{k}': v for k, v in state_dict.items()}
    return state_dict


def test():
    b, h, w, c = shape = 1, 56, 56, 96

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

    conv_jax = ConvNeXtStage(c, out_c,ls_init_value=None)

    conv_jax_params = conv_jax.init(rng, x)['params']
    # print(flax.traverse_util.flatten_dict(conv_jax_params, sep='.').keys())
    print(conv_jax_params.keys())

    # model_torch = ConvNeXtBlock(c, out_c)

    out_jax = conv_jax.apply({'params': conv_jax_params}, x, )  #method=ConvNextBlock.test
    out_jax_np = np.array(out_jax)

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = timm.models.convnext.ConvNeXtStage(c, out_c, kernel_size=kernel_size, stride=stride, ls_init_value=None)
    print(model_torch.state_dict().keys())

    # print(conv_torch.weight.shape)

    state_dict = convert_flax_to_torch_conv_next_stage(conv_jax_params, sep='')

    model_torch.load_state_dict(state_dict)

    # while True:
    #     padding

    # print(conv_torch.weight.shape, conv_torch.state_dict())
    #
    # while True:
    #     padding
    out_torch = model_torch(x_torch)

    out_torch_np = out_torch.detach().numpy()
    out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)







def convert_torch_to_flax_conv_next_stage(torch_params, prefix='', sep='.'):
    blocks = torch_params.pop('blocks')
    for k, v in blocks.items():
        torch_params[f'blocks_{k}'] = v

    # print(torch_params.keys(),torch_params['downsample'].keys())
    # while True:
    #     1

    state_dict = {}
    if 'downsample' in torch_params:
        down_sample_layer_norm=convert_torch_to_flax_layer_norm(torch_params['downsample']['0'],prefix='',sep='')
        down_sample_conv = convert_torch_to_flax_conv(torch_params['downsample']['1'], prefix='',sep='')

        down_sample={'layers_0':down_sample_layer_norm,'layers_1':down_sample_conv}
        state_dict['downsample']=down_sample


        # down_sample_layer_norm = convert_flax_to_torch_layer_norm(flax_params['downsample']['layers_0'],
        #                                                           prefix='downsample.0')
        # down_sample_conv = convert_flax_to_torch_conv(flax_params['downsample']['layers_1'], prefix='downsample.1')
        #
        # state_dict = state_dict | {**down_sample_conv, **down_sample_layer_norm}

    i = 0
    while f'blocks_{i}' in torch_params:
        # state_dict = state_dict | convert_flax_to_torch_conv_next_block(flax_params[f'blocks_{i}'],
        #                                                                 prefix=f'blocks.{i}', sep='.')

        state_dict[f'blocks_{i}']=convert_torch_to_flax_conv_next_block(torch_params[f'blocks_{i}'],prefix='',sep='')
        i += 1


    state_dict = {f'{prefix}{sep}{k}': v for k, v in state_dict.items()}
    return state_dict


def test_torch():
    b, h, w, c = shape = 1, 4, 4, 64

    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)

    out_c = c*2
    kernel_size = 7
    padding = 3
    stride = 2
    use_bias = True

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = timm.models.convnext.ConvNeXtStage(c, out_c, kernel_size=kernel_size, stride=stride, ls_init_value=None,norm_layer= LayerNorm2d)
    print(model_torch.state_dict().keys())


    out_torch = model_torch(x_torch)

    # x = np.asarray(jax.random.normal(rng, shape))

    model_jax = ConvNeXtStage(c, out_c,ls_init_value=None)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    model_jax_params = convert_torch_to_flax_conv_next_stage(params, sep='')
    # print(params)

    out_jax = model_jax.apply({'params': model_jax_params}, x)
    out_jax_np = np.array(out_jax)

    out_torch_np = out_torch.detach().numpy()
    out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)


if __name__ == "__main__":
    test()
