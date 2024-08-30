import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
import einops

from models.convnext import Mlp

jax.config.update('jax_platform_name', 'cpu')





def convert_flax_to_torch_fc(flax_params, prefix='', sep='.', ):
    state_dict = {f'{prefix}{sep}weight': flax_params['kernel'].transpose(1, 0),
                  f'{prefix}{sep}bias': flax_params['bias']}
    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    return state_dict


def convert_flax_to_torch_mlp(flax_params, prefix='', sep='.', ):
    fc1 = convert_flax_to_torch_fc(flax_params['fc1'], prefix='fc1')
    fc2 = convert_flax_to_torch_fc(flax_params['fc2'], prefix='fc2')

    state_dict = {**fc1, **fc2}
    state_dict = {f'{prefix}{sep}{k}': v for k, v in state_dict.items()}
    # print(state_dict)
    #
    # while True:
    #     pass
    return state_dict


def test():
    b, h, w, c = shape = 1, 1, 1,1024

    rng = jax.random.PRNGKey(0)
    # x = jnp.ones(shape)
    x = np.asarray(jax.random.normal(rng, shape))
    #

    out_c = 256
    kernel_size = 3
    padding = 1
    stride = 1
    use_bias = True
    mlp_ratio = 4

    model_jax = Mlp(mlp_ratio * c, c)

    model_jax_params = model_jax.init(rng, x)['params']
    print(model_jax_params)

    out_jax = model_jax.apply({'params': model_jax_params}, x)
    out_jax_np = np.array(out_jax)

    from timm.layers import Mlp as timm_mlp
    model_torch = timm_mlp(c, mlp_ratio * c, c, act_layer=tnn.GELU)
    print(model_torch.state_dict().keys())

    # while True:
    #     padding

    # state_dict = {'weight': conv_jax_params['kernel'].transpose(3, 2, 0, 1),'bias':conv_jax_params['bias']}
    # state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    state_dict = convert_flax_to_torch_mlp(model_jax_params, sep='')

    # while True:
    #     pass

    model_torch.load_state_dict(state_dict)
    x_torch = torch.from_numpy(np.array(x))
    # x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')
    out_torch = model_torch(x_torch)

    out_torch_np = out_torch.detach().numpy()
    # out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=5)











def convert_torch_to_flax_fc(torch_params, prefix='', sep='.', ):

    state_dict = {f'{prefix}{sep}kernel': torch_params['weight'].transpose(1, 0),
                  f'{prefix}{sep}bias': torch_params['bias']}
    # print('\n'*5)
    # print(state_dict)
    # while True:
    #     1
    # state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    return state_dict


def convert_torch_to_flax_mlp(torch_params, prefix='', sep='.', ):
    fc1 = convert_torch_to_flax_fc(torch_params['fc1'], prefix='',sep='')
    fc2 = convert_torch_to_flax_fc(torch_params['fc2'], prefix='',sep='')

    state_dict={'fc1':fc1,'fc2':fc2}
    return state_dict


def test_torch():
    b, h, w, c = shape = 1, 1, 1,1024
    from timm.layers import Mlp as timm_mlp
    mlp_ratio=4


    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)
    x_torch = torch.from_numpy(np.array(x))
    # x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = timm_mlp(c, mlp_ratio * c, c, act_layer=tnn.GELU)

    print(model_torch.state_dict().keys())
    out_torch = model_torch(x_torch)

    # x = np.asarray(jax.random.normal(rng, shape))

    model_jax = Mlp(mlp_ratio * c, c)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")


    print(params.keys())

    model_jax_params = convert_torch_to_flax_mlp(params, sep='')
    print(params)

    out_jax = model_jax.apply({'params': model_jax_params}, x)
    out_jax_np = np.array(out_jax)

    out_torch_np = out_torch.detach().numpy()
    # out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=5)



if __name__ == "__main__":
    # test()
    test_torch()