import numpy as np
import timm
import flax
import torch
from flax.configurations import flax_preserve_adopted_names
from torch.nn.init import normal


def convert_flax_to_torch_conv(flax_params, prefix='', sep='.'):
    """
    Converts Flax convolutional layer parameters to a PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters with keys 'kernel' and 'bias'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Check if the kernel is for a convolutional layer (4D)
    if 'kernel' in flax_params:
        # Flax convolution kernel shape: [H, W, in_channels, out_channels]
        # PyTorch convolution kernel shape: [out_channels, in_channels, H, W]
        state_dict[f'{prefix}{sep}weight'] = flax_params['kernel'].transpose(3, 2, 0, 1)
    else:
        raise ValueError("Expected a 4D kernel for convolutional conversion, but found a different shape.")

    # Convert Flax 'bias' (if it exists) to PyTorch 'bias'
    if 'bias' in flax_params:
        state_dict[f'{prefix}{sep}bias'] = flax_params['bias']

    # Convert all parameters to PyTorch tensors
    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict




def convert_flax_to_torch_layer_norm(flax_params, prefix='', sep='.'):
    """
    Convert Flax LayerNorm parameters to a PyTorch-compatible state dict.

    Args:
        flax_params (dict): Flax LayerNorm parameters with keys 'scale' and 'bias'.
        prefix (str): Prefix for the parameter names in the resulting state dict.
        sep (str): Separator for joining prefix and keys.

    Returns:
        dict: State dict compatible with PyTorch nn.LayerNorm.
    """
    state_dict = {}

    # Convert Flax 'scale' (equivalent to PyTorch 'weight')
    if 'scale' in flax_params:
        state_dict[f'{prefix}{sep}weight'] = torch.tensor(flax_params['scale'])

    # Convert Flax 'bias' (equivalent to PyTorch 'bias')
    if 'bias' in flax_params:
        state_dict[f'{prefix}{sep}bias'] = torch.tensor(flax_params['bias'])

    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict




def convert_flax_to_torch_linear(flax_params, prefix='', sep='.'):
    """
    Converts Flax layer parameters (linear or convolutional) to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters with keys 'kernel' and 'bias'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Check if the kernel is for a convolutional layer (4D)
    if 'kernel' in flax_params and len(flax_params['kernel'].shape) == 4:
        # Flax convolution kernel shape: [H, W, in_channels, out_channels]
        # PyTorch convolution kernel shape: [out_channels, in_channels, H, W]
        state_dict[f'{prefix}{sep}weight'] = flax_params['kernel'].transpose(3, 2, 0, 1)
    elif 'kernel' in flax_params:  # Dense (linear) layer case
        # Flax dense kernel shape: [in_features, out_features]
        # PyTorch linear weight shape: [out_features, in_features]
        state_dict[f'{prefix}{sep}weight'] = flax_params['kernel'].T

    # Convert Flax 'bias' (if it exists) to PyTorch 'bias'
    if 'bias' in flax_params:
        state_dict[f'{prefix}{sep}bias'] = flax_params['bias']

    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict



def convert_flax_to_torch_stem(flax_params, prefix='', sep='.'):
    """
    Converts Flax Stem module parameters (Conv2D + optional normalization layer)
    to a PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters.
            Expected keys: 'conv' and optionally 'norm'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Convert convolution parameters using the predefined function
    if 'conv' in flax_params:
        conv_state_dict = convert_flax_to_torch_conv(flax_params['conv'], prefix=f'{prefix}{sep}conv', sep='.')
        state_dict.update(conv_state_dict)

    # Convert normalization layer parameters (if present) using the predefined function
    if 'norm' in flax_params:
        norm_state_dict = convert_flax_to_torch_layer_norm(flax_params['norm'], prefix=f'{prefix}{sep}norm', sep='.')
        state_dict.update(norm_state_dict)

    # Convert all parameters to PyTorch tensors
    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict



def convert_flax_to_torch_star_relu(flax_params, prefix='', sep='.'):
    """
    Converts Flax StarReLU module parameters to a PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters.
            Expected keys: 'scale' and 'bias'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Convert scale parameter
    if 'scale' in flax_params:
        state_dict[f'{prefix}{sep}scale'] = flax_params['scale']

    # Convert bias parameter
    if 'bias' in flax_params:
        state_dict[f'{prefix}{sep}bias'] = flax_params['bias']

    # Convert all parameters to PyTorch tensors
    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict


def convert_flax_to_torch_scale(flax_params, prefix='', sep='.'):
    """
    Converts Flax Scale module parameters to a PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters.
            Expected keys: 'scale'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Convert scale parameter
    if 'scale' in flax_params:
        state_dict[f'{prefix}{sep}scale'] = flax_params['scale']

    # Convert all parameters to PyTorch tensors
    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict

def convert_flax_to_torch_mlp(flax_params, prefix='', sep='.'):
    """
    Converts Flax MLP module parameters to a PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters.
            Expected keys: 'fc1.kernel', 'fc1.bias', 'fc2.kernel', 'fc2.bias'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """

    fc1=convert_flax_to_torch_linear(flax_params['fc1'],prefix='fc1')
    fc2 = convert_flax_to_torch_linear(flax_params['fc2'], prefix='fc2')
    state_dict = {**fc1,**fc2}

    if 'act' in flax_params:
        state_dict.update(convert_flax_to_torch_star_relu(flax_params['act'],prefix='act'))

    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict




def convert_flax_to_torch_sepconv(flax_params, prefix='', sep='.'):
    """
    Converts Flax SepConv (Inverted Separable Convolution) parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters.
            Expected keys: 'pwconv1.kernel', 'pwconv1.bias', 'dwconv.kernel', 'dwconv.bias', 'pwconv2.kernel', 'pwconv2.bias'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """


    pw_conv1=convert_flax_to_torch_conv(flax_params['pwconv1'],prefix='pwconv1')
    pw_conv2 = convert_flax_to_torch_conv(flax_params['pwconv2'], prefix='pwconv2')
    dwconv=convert_flax_to_torch_conv(flax_params['dwconv'], prefix='dwconv')

    state_dict = {**pw_conv1,**pw_conv2,**dwconv}

    if 'act1' in flax_params:
        act1 = convert_flax_to_torch_star_relu(flax_params['act1'], prefix='act1')
        state_dict.update(act1)
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    return state_dict


def convert_flax_to_torch_metafomer_block(flax_params, prefix='', sep='.'):
    """
    Converts Flax MetaFormerBlock parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters, expected keys like 'norm1', 'token_mixer', 'mlp', etc.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """



    norm1 = convert_flax_to_torch_layer_norm(flax_params['norm1'],prefix='norm1')
    norm2 = convert_flax_to_torch_layer_norm(flax_params['norm2'],prefix='norm2')
    mlp=convert_flax_to_torch_mlp(flax_params['mlp'],prefix='mlp')

    if 'pwconv1' in flax_params['token_mixer']:
        token_mixer=convert_flax_to_torch_sepconv(flax_params['token_mixer'],prefix='token_mixer')
    elif 'qkv' in flax_params['token_mixer']:
        token_mixer=convert_flax_to_torch_attention(flax_params['token_mixer'],prefix='token_mixer')
    else:
        raise NotImplemented()

    state_dict={**norm1,**norm2,**mlp,**token_mixer}


    if 'res_scale1' in flax_params:
        state_dict.update(convert_flax_to_torch_scale(flax_params['res_scale1'],prefix='res_scale1'))

    if 'res_scale2' in flax_params:
        state_dict.update(convert_flax_to_torch_scale(flax_params['res_scale2'],prefix='res_scale2'))


    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict


def convert_flax_to_torch_downsampling(flax_params, prefix='', sep='.'):
    """
    Converts Flax Downsampling parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters, expected keys like 'conv', 'norm'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Convert Flax 'norm' LayerNorm parameters (if norm_layer is provided)
    if 'norm' in flax_params:
       norm=convert_flax_to_torch_layer_norm(flax_params['norm'],prefix='norm')
       state_dict.update(norm)

    # Convert 'conv' parameters (kernel and bias)
    state_dict.update(convert_flax_to_torch_conv(flax_params['conv'],prefix='conv'))

    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict



def convert_flax_to_torch_attention(flax_params, prefix='', sep='.'):
    """
    Converts Flax Downsampling parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters, expected keys like 'conv', 'norm'.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}
    state_dict.update(convert_flax_to_torch_linear(flax_params['qkv'],prefix='qkv'))
    state_dict.update(convert_flax_to_torch_linear(flax_params['proj'],prefix='proj'))





    if 'LayerNorm_0' in flax_params:
        state_dict.update(convert_flax_to_torch_layer_norm(flax_params['LayerNorm_0'], prefix='q_norm'))
        state_dict.update(convert_flax_to_torch_layer_norm(flax_params['LayerNorm_1'], prefix='k_norm'))

    # print(state_dict.keys())
    # while True:
    #     pass


    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict


def convert_flax_to_torch_metaformer_stage(flax_params, prefix='', sep='.'):
    """
    Converts Flax MetaFormerStage parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters for MetaFormerStage.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary.
    """
    state_dict = {}

    # Convert downsample parameters (if Downsampling layer exists)
    if 'downsample' in flax_params:
        state_dict.update(convert_flax_to_torch_downsampling(flax_params['downsample'], prefix=f'downsample'))
        flax_params.pop('downsample')

    # Convert MetaFormerBlocks parameters
    for i, block_params in enumerate(flax_params):
        state_dict.update(convert_flax_to_torch_metafomer_block(flax_params[f'MetaFormerBlock_{i}'], prefix=f'blocks.{i}'))

    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict


def convert_flax_to_torch_mlp_head(flax_params, prefix='', sep='.'):
    """
    Converts Flax MlpHead parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters for MlpHead.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary for MlpHead.
    """
    state_dict = {}

    # Convert first fully connected layer (fc1)
    if 'fc1' in flax_params:
        state_dict.update(convert_flax_to_torch_linear(flax_params['fc1'], prefix=f'fc1'))

    # Convert activation function (act_layer) if necessary, but no parameters to convert

    # Convert normalization layer (norm)
    if 'norm' in flax_params:
        state_dict.update(convert_flax_to_torch_layer_norm(flax_params['norm'], prefix=f'norm'))

    # Convert dropout parameters (head_dropout)
    # No parameters to convert here, PyTorch will handle this in the model

    # Convert second fully connected layer (fc2)
    if 'fc2' in flax_params:
        state_dict.update(convert_flax_to_torch_linear(flax_params['fc2'], prefix=f'fc2'))

    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    return state_dict


def convert_flax_to_torch_meta_former(flax_params, prefix='', sep='.'):
    """
    Converts Flax MlpHead parameters to PyTorch-compatible state dictionary.

    Args:
        flax_params (dict): Dictionary containing Flax parameters for MlpHead.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: PyTorch-compatible state dictionary for MlpHead.
    """

    state_dict = {}
    # Convert downsample parameters (if Downsampling layer exists)
    state_dict.update(convert_flax_to_torch_stem(flax_params['stem'], prefix=f'stem'))
    flax_params.pop('stem')

    if 'out_norm' in flax_params:
        norm=convert_flax_to_torch_layer_norm(flax_params['out_norm'], prefix=f'head.norm')
        flax_params.pop('out_norm')
        state_dict.update(norm)

    if 'fc' in flax_params:
        fc = convert_flax_to_torch_mlp_head(flax_params['fc'], prefix='head.fc')
        flax_params.pop('fc')
        state_dict.update(fc)

    # Convert MetaFormerBlocks parameters
    for i, stage_params in enumerate(flax_params):
        state_dict.update(convert_flax_to_torch_metaformer_stage(flax_params[f'MetaFormerStage_{i}'], prefix=f'{prefix}{sep}stages.{i}'))

    # Convert all parameters to PyTorch tensors
    state_dict = {f'{prefix}{sep}{k}': torch.tensor(np.asarray(v)) for k, v in state_dict.items()}


    return state_dict
