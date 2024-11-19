import timm
import flax

def convert_torch_to_flax_linear(torch_params,  prefix='', sep=''):
    """
    Converts PyTorch layer weights and biases (in NumPy format) to Flax-compatible format.

    Args:
        torch_params (numpy.ndarray): The weight matrix of the layer.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        torch_params: A dictionary with Flax-compatible weights and biases.
    """
    flax_params = {}
    if len(torch_params['weight'].shape) == 4:  # Convolution
        flax_params[f'{prefix}{sep}kernel'] = torch_params['weight'].transpose(2, 3, 1, 0)
    else:  # Dense
        flax_params[f'{prefix}{sep}kernel'] = torch_params['weight'].T
    if 'bias' in torch_params:
        flax_params[f'{prefix}{sep}bias'] = torch_params['bias']


    return flax_params

def load_pretrain(pretrained_model='caformer_b36.sail_in1k', default_params=None):
    model_torch = timm.create_model(pretrained_model, pretrained=True)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    print(params['stem'].keys())


def convert_torch_to_flax_mlp(torch_params, prefix='', sep=''):
    """
    Converts PyTorch MLP parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): PyTorch state_dict for MLP in NumPy format.
        prefix (str): Optional prefix for parameter names.
        sep (str): Separator for parameter names.

    Returns:
        dict: Flax-compatible parameter dictionary.
    """

    flax_params = {f'fc1': convert_torch_to_flax_linear(torch_params['fc1']),
                   f'fc2': convert_torch_to_flax_linear(torch_params['fc2'])}
    # First linear/convolutional layer

    # Optional normalization layer
    if 'norm.weight' in torch_params and 'norm.bias' in torch_params:
        norm_prefix = f'{prefix}{sep}norm'
        flax_params[f'{norm_prefix}{sep}scale'] = torch_params['norm.weight']
        flax_params[f'{norm_prefix}{sep}bias'] = torch_params['norm.bias']

    if 'act' in torch_params:
        flax_params['act'] = convert_torch_to_flax_star_relu(torch_params['act'])

    return flax_params


def convert_torch_to_flax_conv(torch_params, prefix='', sep='', ):
    state_dict = {f'{prefix}{sep}kernel': torch_params['weight'].transpose(2, 3, 1, 0), }

    if 'bias' in torch_params:
        state_dict[f'{prefix}{sep}bias'] = torch_params['bias']
    # state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    return state_dict


def convert_torch_to_flax_layer_norm(torch_params, prefix='', sep='', ):
    flax_params = {f'{prefix}{sep}scale': torch_params['weight'], }

    if 'bias' in torch_params:
        flax_params[f'{prefix}{sep}bias'] = torch_params['bias']

    return flax_params


def convert_torch_to_flax_stem(torch_params):
    flax_params = dict()

    flax_params['conv'] = convert_torch_to_flax_conv(torch_params['conv'])
    flax_params['norm'] = convert_torch_to_flax_layer_norm(torch_params['norm'])
    return flax_params


def convert_torch_to_flax_star_relu(torch_params, prefix=''):
    state_dict = dict()
    # Convert scale parameter
    state_dict[f'{prefix}scale'] = torch_params['scale']
    # Convert bias parameter
    state_dict[f'{prefix}bias'] = torch_params['bias']
    return state_dict


def convert_torch_to_flax_sep_conv(torch_params, prefix='', sep=''):
    """
    Converts PyTorch SepConv parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): Dictionary containing PyTorch state_dict of SepConv.
        prefix (str): Optional prefix for the Flax parameter names.
        sep (str): Separator for the Flax parameter names.

    Returns:
        dict: Flax-compatible parameters for SepConv.
    """
    flax_params = {}

    # Convert pwconv1
    pwconv1_key = f'{prefix}{sep}pwconv1'
    flax_params['pwconv1'] = convert_torch_to_flax_conv(torch_params['pwconv1'], )
    flax_params['dwconv'] = convert_torch_to_flax_conv(torch_params['dwconv'], )
    flax_params['pwconv2'] = convert_torch_to_flax_conv(torch_params['pwconv2'], )

    if 'act1' in torch_params:
        flax_params['act1'] = convert_torch_to_flax_star_relu(torch_params['act1'])

    return flax_params


def convert_torch_to_flax_scale(torch_params, prefix='', sep=''):
    """
    Converts PyTorch Scale parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): Dictionary containing PyTorch state_dict of Scale.
        prefix (str): Optional prefix for the Flax parameter names.
        sep (str): Separator for the Flax parameter names.

    Returns:
        dict: Flax-compatible parameters for Scale.
    """
    flax_params = {
        f'{prefix}{sep}scale': torch_params['scale']  # Scale parameter (NumPy array)
    }
    return flax_params



def convert_torch_to_flax_attention(torch_params, prefix='', sep=''):
    """
    Converts PyTorch Scale parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): Dictionary containing PyTorch state_dict of Scale.
        prefix (str): Optional prefix for the Flax parameter names.
        sep (str): Separator for the Flax parameter names.

    Returns:
        dict: Flax-compatible parameters for Scale.
    """
    flax_params = {
        f'{prefix}{sep}qkv': convert_torch_to_flax_linear(torch_params['qkv']  ),
        f'{prefix}{sep}proj': convert_torch_to_flax_linear(torch_params['proj']),
    }


    # print(torch_params['qkv']['weight'].shape,flax_params['qkv']['kernel'].shape)
    # while True:
    #     pass

    return flax_params



def convert_torch_to_flax_meta_former_block(torch_params, prefix='', sep=''):
    """
    Converts PyTorch Scale parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): Dictionary containing PyTorch state_dict of Scale.
        prefix (str): Optional prefix for the Flax parameter names.
        sep (str): Separator for the Flax parameter names.

    Returns:
        dict: Flax-compatible parameters for Scale.
    """

    flax_params = {
        'norm1': convert_torch_to_flax_layer_norm(torch_params['norm1']),
        'norm2': convert_torch_to_flax_layer_norm(torch_params['norm2']),
        'mlp': convert_torch_to_flax_mlp(torch_params['mlp'])
    }




    if 'res_scale1' in torch_params:
        flax_params['res_scale1']=convert_torch_to_flax_scale(torch_params['res_scale1'])


    if 'res_scale2' in torch_params:
        flax_params['res_scale2']=convert_torch_to_flax_scale(torch_params['res_scale2'])


    if 'token_mixer' in torch_params:

        if 'qkv' in torch_params['token_mixer']:
            flax_params['token_mixer'] = convert_torch_to_flax_attention(torch_params['token_mixer'])
        else:
            flax_params['token_mixer'] = convert_torch_to_flax_sep_conv(torch_params['token_mixer'])

    return flax_params


def convert_torch_to_flax_meta_former_stage(torch_params, prefix='', sep=''):
    """
    Converts PyTorch Scale parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): Dictionary containing PyTorch state_dict of Scale.
        prefix (str): Optional prefix for the Flax parameter names.
        sep (str): Separator for the Flax parameter names.

    Returns:
        dict: Flax-compatible parameters for Scale.
    """

    flax_params = dict()

    if 'downsample' in torch_params:
        flax_params['downsample']= convert_torch_to_flax_stem(torch_params['downsample'])

    blocks = torch_params['blocks']

    for i in blocks.keys():
        flax_params[f'MetaFormerBlock_{i}'] = convert_torch_to_flax_meta_former_block(blocks[i])

    return flax_params


def convert_torch_to_flax_mlp_head(torch_params, prefix='', sep=''):
    """
    Convert PyTorch MlpHead weights to Flax weights.
    torch_params is a dictionary of NumPy arrays representing the PyTorch weights.
    """
    flax_params = {
        'fc1': convert_torch_to_flax_linear(torch_params['fc1']),
        'fc2': convert_torch_to_flax_linear(torch_params['fc2']),
        'norm': convert_torch_to_flax_layer_norm(torch_params['norm']),

    }
    # Remove any None values for missing parameters
    return flax_params


def convert_torch_to_flax_cls_head(torch_params, prefix='', sep=''):
    """
    Convert PyTorch MlpHead weights to Flax weights.
    torch_params is a dictionary of NumPy arrays representing the PyTorch weights.
    """
    flax_params = {
        'norm': convert_torch_to_flax_layer_norm(torch_params['norm']),
        'fc': convert_torch_to_flax_mlp_head(torch_params['fc']),

    }
    # Remove any None values for missing parameters
    return flax_params


def convert_torch_to_flax_meta_former(torch_params, prefix='', sep=''):
    """
    Converts PyTorch Scale parameters (NumPy format) to Flax-compatible format.

    Args:
        torch_params (dict): Dictionary containing PyTorch state_dict of Scale.
        prefix (str): Optional prefix for the Flax parameter names.
        sep (str): Separator for the Flax parameter names.

    Returns:
        dict: Flax-compatible parameters for Scale.
    """

    flax_params = dict()
    flax_params['stem'] = convert_torch_to_flax_stem(torch_params['stem'])

    # flax_params['stages']=convert_torch_to_flax_meta_former_stage(torch_params['stages'])
    stages = torch_params['stages']
    for i in stages.keys():
        flax_params[f'MetaFormerStage_{i}'] = convert_torch_to_flax_meta_former_stage(stages[i])

    if 'head' in torch_params:
        flax_params['out_norm']= convert_torch_to_flax_layer_norm(torch_params['head']['norm'])
        flax_params['fc']= convert_torch_to_flax_mlp_head(torch_params['head']['fc'])


    return flax_params