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
from .check_layernorm import convert_flax_to_torch_layer_norm,convert_torch_to_flax_layer_norm
from .check_mlp import convert_flax_to_torch_mlp,convert_torch_to_flax_mlp
from models.convnext import Identity, ConvNeXtBlock

jax.config.update('jax_platform_name', 'cpu')


class ModelTorch(timm.models.convnext.ConvNeXtBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def convert_flax_to_torch_conv_next_block(flax_params, prefix='', sep='.'):
    conv_dw = convert_flax_to_torch_conv(flax_params=flax_params['conv_dw'], prefix='conv_dw')
    norm = convert_flax_to_torch_layer_norm(flax_params=flax_params['norm'], prefix='norm')
    mlp = convert_flax_to_torch_mlp(flax_params=flax_params['mlp'], prefix='mlp')

    state_dict = {**conv_dw, **norm, **mlp}

    if 'gamma' in flax_params:
        state_dict.update({'gamma': torch.from_numpy(np.array(flax_params['gamma']))})

    # state_dict = {**conv_dw, }
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

    conv_jax = ConvNeXtBlock(c, out_c)

    conv_jax_params = conv_jax.init(rng, x)['params']
    # print(flax.traverse_util.flatten_dict(conv_jax_params, sep='.').keys())
    print(conv_jax_params.keys())

    # model_torch = ConvNeXtBlock(c, out_c)

    out_jax = conv_jax.apply({'params': conv_jax_params}, x, )
    out_jax_np = np.array(out_jax)

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = ModelTorch(c, out_c, kernel_size=kernel_size, stride=stride, ls_init_value=1e-6)
    print(model_torch.state_dict().keys())

    # print(conv_torch.weight.shape)

    state_dict = convert_flax_to_torch_conv_next_block(conv_jax_params, sep='')

    model_torch.load_state_dict(state_dict)

    # print(conv_torch.weight.shape, conv_torch.state_dict())
    #
    # while True:
    #     padding
    out_torch = model_torch(x_torch)

    out_torch_np = out_torch.detach().numpy()
    out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=5)









def convert_torch_to_flax_conv_next_block(torch_params, prefix='', sep='.'):

    conv_dw = convert_torch_to_flax_conv(torch_params['conv_dw'], prefix='',sep='')
    norm = convert_torch_to_flax_layer_norm(torch_params['norm'], prefix='',sep='')
    mlp = convert_torch_to_flax_mlp(torch_params['mlp'], prefix='',sep='')

    state_dict = {'conv_dw':conv_dw, 'norm':norm, 'mlp':mlp}

    if 'gamma' in torch_params:
        state_dict.update({'gamma':np.array(torch_params['gamma'])})

    # state_dict = {**conv_dw, }
    state_dict = {f'{prefix}{sep}{k}': v for k, v in state_dict.items()}
    return state_dict


def test_torch():
    b, h, w, c = shape = 1, 4, 4, 64

    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)

    out_c = c
    kernel_size = 7
    padding = 3
    stride = 1
    use_bias = True

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = ModelTorch(c, out_c, kernel_size=kernel_size, stride=stride, ls_init_value=None)
    print(model_torch.state_dict().keys())
    out_torch = model_torch(x_torch)

    # x = np.asarray(jax.random.normal(rng, shape))

    model_jax = ConvNeXtBlock(c, out_c,ls_init_value=None)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    model_jax_params = convert_torch_to_flax_conv_next_block(params, sep='')
    print(params)

    out_jax = model_jax.apply({'params': model_jax_params}, x)
    out_jax_np = np.array(out_jax)

    out_torch_np = out_torch.detach().numpy()
    out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)



if __name__ == "__main__":
    test()
