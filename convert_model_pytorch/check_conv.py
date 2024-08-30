import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
import einops

jax.config.update('jax_platform_name', 'cpu')


def convert_flax_to_torch_conv(flax_params,prefix='', sep='.', ):
    state_dict = {f'{prefix}{sep}weight': flax_params['kernel'].transpose(3, 2, 0, 1),
                  f'{prefix}{sep}bias': flax_params['bias']}
    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    return state_dict

def test():
    b, h, w, c = shape = 1, 4, 4, 64

    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)

    out_c = c
    kernel_size = 7
    padding = 3
    stride = 1
    use_bias = True

    conv_jax = nn.Conv(out_c, (kernel_size, kernel_size),
                       use_bias=use_bias,
                       precision='highest',
                       padding=[(padding, padding), (padding, padding)], strides=(stride, stride))
    conv_jax_params = conv_jax.init(rng, x)['params']

    print(conv_jax_params['kernel'].shape)

    out_jax = conv_jax.apply({'params': conv_jax_params}, x)
    out_jax_np = np.array(out_jax)

    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    conv_torch = tnn.Conv2d(c, out_c, kernel_size=kernel_size, bias=use_bias, padding=padding, stride=stride)

    # state_dict = {'weight': conv_jax_params['kernel'].transpose(3, 2, 0, 1),'bias':conv_jax_params['bias']}
    # state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    state_dict = convert_flax_to_torch_conv(conv_jax_params,sep='')

    conv_torch.load_state_dict(state_dict)
    # print(conv_torch.weight.shape, conv_torch.state_dict())
    #
    # while True:
    #     padding
    out_torch = conv_torch(x_torch)

    out_torch_np = out_torch.detach().numpy()
    out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)





def convert_torch_to_flax_conv(torch_params,prefix='', sep='.', ):
    state_dict = {f'{prefix}{sep}kernel': torch_params['weight'].transpose(2, 3, 1, 0),
                  f'{prefix}{sep}bias': torch_params['bias']}
    # state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    return state_dict

def test_torch_to_jax():
    b, h, w, c = shape = 1, 4, 4, 64

    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)

    out_c = c
    kernel_size = 7
    padding = 3
    stride = 1
    use_bias = True



    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape)
    x_torch = torch.from_numpy(np.array(x))
    x_torch = einops.rearrange(x_torch, 'b h w c->b c h w')

    model_torch = tnn.Conv2d(c, out_c, kernel_size=kernel_size, bias=use_bias, padding=padding, stride=stride)
    print(model_torch.state_dict().keys())
    out_torch = model_torch(x_torch)

    # x = np.asarray(jax.random.normal(rng, shape))

    model_jax =  nn.Conv(out_c, (kernel_size, kernel_size),
                       use_bias=use_bias,
                       # precision='highest',
                       padding=[(padding, padding), (padding, padding)], strides=(stride, stride))
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    model_jax_params = convert_torch_to_flax_conv(params, sep='')
    print(params)

    out_jax = model_jax.apply({'params': model_jax_params}, x)
    out_jax_np = np.array(out_jax)

    out_torch_np = out_torch.detach().numpy()
    out_torch_np = einops.rearrange(out_torch_np, 'b c h w ->b h w c')
    print(out_torch_np.shape)
    print(out_torch_np - out_jax_np)

    np.testing.assert_almost_equal(out_torch_np, out_jax_np, decimal=6)


if __name__ == "__main__":
    test_torch_to_jax()
