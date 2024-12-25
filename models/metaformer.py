from dataclasses import field
from functools import partial
from typing import Callable, Optional, Sequence, Union, Any

import einops
import jax.experimental.pallas.ops.tpu.flash_attention
import numpy as np

from pre_define import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .layers import Mlp, DropPath, Dense, Conv

use_fast_variance = True


import flax.linen as nn
import jax.numpy as jnp


dtype=jnp.bfloat16

class SquaredReLU(nn.Module):
    """
    Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    @nn.compact
    def __call__(self, x):
        relu = nn.relu(x)  # Flax has an in-built ReLU function
        return jnp.square(relu)  # Squaring the output of the ReLU activation



class Identity(nn.Module):
    def __call__(self, x,det=True):
        return x


class Stem(nn.Module):
    """
    Stem implemented by a layer of convolution.
    Conv2d params constant across all models.
    """
    out_channels: int
    norm_layer: nn.Module = partial(nn.LayerNorm, use_bias=False, name='norm', use_fast_variance=use_fast_variance)

    @nn.compact
    def __call__(self, x):
        x = Conv(
            features=self.out_channels,
            kernel_size=(7, 7),
            strides=(4, 4),
            padding=((2, 2), (2, 2)),
            name='conv',dtype=dtype
        )(x)
        if self.norm_layer:
            x = self.norm_layer(name='norm',dtype=dtype)(x)
        return x

#
class StarReLU(nn.Module):
    scale_value: float = 1.0
    bias_value: float = 0.0
    scale_learnable: bool = True
    bias_learnable: bool = True

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', lambda rng, shape: jnp.full(shape, self.scale_value), (1,))
        bias = self.param('bias', lambda rng, shape: jnp.full(shape, self.bias_value), (1,))

        if not self.scale_learnable:
            scale = jnp.array(scale).at[()].set(self.scale_value)
        if not self.bias_learnable:
            bias = jnp.array(bias).at[()].set(self.bias_value)

        return scale.astype(x.dtype) * nn.relu(x) ** 2 + bias.astype(x.dtype)



# class StarReLU(nn.Module):
#     scale_value: float = 1.0
#     bias_value: float = 0.0
#     scale_learnable: bool = False
#     bias_learnable: bool = False
#
#     @nn.compact
#     def __call__(self, x):
#         return nn.relu(x)

class Scale(nn.Module):
    """
    Flax implementation of Scale: Element-wise scaling of input.
    """
    dim: int
    init_value: float = 1.0
    trainable: bool = True
    use_nchw: bool = True

    @nn.compact
    def __call__(self, x):
        # Initialize scale parameter
        scale = self.param(
            'scale',
            lambda rng: jnp.ones((self.dim,)) * self.init_value,  # Flax expects 1D array
            # mutable=True
        )

        if not self.trainable:
            scale = jnp.array(scale, dtype=x.dtype)

        # Reshape scale to match the target shape
        shape = (1, 1, self.dim) if self.use_nchw else (self.dim,)
        scale = scale.reshape(shape)

        # Perform element-wise scaling
        return x * scale.astype(x.dtype)






class SepConv(nn.Module):
    """
    Flax implementation of Inverted Separable Convolution (MobileNetV2 style).
    """
    dim: int
    expansion_ratio: float = 2.0
    act1_layer: Callable[..., nn.Module] = StarReLU  # Default to ReLU; replaceable
    act2_layer: Callable[..., nn.Module] = Identity
    bias: bool = False
    kernel_size: int = 7
    padding: int = 3

    @nn.compact
    def __call__(self, x):
        mid_channels = int(self.expansion_ratio * self.dim)

        # Pointwise Convolution 1
        pwconv1 = Conv(
            features=mid_channels, kernel_size=(1, 1), use_bias=self.bias, name='pwconv1',
            dtype = dtype
        )(x)
        x = self.act1_layer(name='act1')(pwconv1)

        # Depthwise Convolution
        dwconv = Conv(
            features=mid_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            feature_group_count=mid_channels,  # Depthwise
            use_bias=self.bias,
            name='dwconv',dtype = dtype
        )(x)
        x = self.act2_layer()(dwconv)

        # Pointwise Convolution 2
        pwconv2 = Conv(
            features=self.dim, kernel_size=(1, 1), use_bias=self.bias, name='pwconv2',dtype = dtype
        )(x)

        return pwconv2


class Attention(nn.Module):
    dim: int
    head_dim: int = 128 #32
    num_heads: int = None
    qkv_bias: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    proj_bias: bool = False
    fused_attn: bool = False  # Assume the use_fused_attn() logic will be passed explicitly

    @nn.compact
    def __call__(self, x,det=True):
        B, N, C = x.shape
        head_dim = self.head_dim
        num_heads = self.num_heads or C // head_dim
        num_heads = max(1, num_heads)
        attention_dim = num_heads * head_dim
        scale = head_dim ** -0.5

        qkv = nn.Dense(attention_dim * 3, use_bias=self.qkv_bias, name="qkv",dtype = dtype)(x)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if N==256:
            x=jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(q,k,v)
        else:
            attn = jnp.einsum("...nd,...md->...nm", q, k) * scale
            attn = nn.softmax(attn, axis=-1)
            x = jnp.einsum("...nm,...md->...nd", attn, v)

        x = x.transpose((0, 2, 1, 3)).reshape(B, N, C)
        x = nn.Dense(C, use_bias=self.proj_bias, name="proj",dtype = dtype)(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic=det)
        return x



class MetaFormerBlock(nn.Module):
    """
    Flax implementation of MetaFormer block.
    """

    dim: int
    token_mixer: Callable[..., nn.Module] = SepConv  # Adjust according to your use
    mlp_act: Callable[..., nn.Module] = StarReLU
    mlp_bias: bool = False
    norm_layer: nn.Module = partial(nn.LayerNorm, use_bias=True,
                                    use_fast_variance=use_fast_variance)  # Default LayerNorm
    proj_drop: float = 0.0
    drop_path: float = 0.0
    use_nchw: bool = True
    layer_scale_init_value: float = None
    res_scale_init_value: float = None

    @nn.compact
    def __call__(self, x,det=True):
        # Layer scale and residual scale initializers
        ls_layer = partial(Scale, dim=self.dim, init_value=self.layer_scale_init_value, use_nchw=self.use_nchw)
        rs_layer = partial(Scale, dim=self.dim, init_value=self.res_scale_init_value, use_nchw=self.use_nchw)

        # Norm1, Token Mixer, Drop Path1, Layer Scale1, Residual Scale1
        norm1 = self.norm_layer(name="norm1")
        token_mixer = self.token_mixer(dim=self.dim, name='token_mixer')
        drop_path1 = DropPath(self.drop_path) if self.drop_path > 0. else Identity()
        layer_scale1 = ls_layer() if self.layer_scale_init_value is not None else Identity()
        res_scale1 = rs_layer(name='res_scale1') if self.res_scale_init_value is not None else Identity()

        # Norm2, MLP, Drop Path2, Layer Scale2, Residual Scale2
        norm2 = self.norm_layer(name="norm2")
        mlp = Mlp(
            self.dim,
            int(4 * self.dim),
            act_layer=self.mlp_act,
            bias=self.mlp_bias,
            drop=self.proj_drop,
            use_conv=self.use_nchw, name='mlp'
        )
        drop_path2 = DropPath(self.drop_path) if self.drop_path > 0. else Identity()
        layer_scale2 = ls_layer() if self.layer_scale_init_value is not None else Identity()
        res_scale2 = rs_layer(name='res_scale2') if self.res_scale_init_value is not None else Identity()


        # First block (Token Mixer + Layer 1 transformations)
        x = res_scale1(x) + layer_scale1(drop_path1(token_mixer(norm1(x)) ,det ))

        # Second block (MLP + Layer 2 transformations)
        x = res_scale2(x) + layer_scale2(drop_path2(mlp(norm2(x),det) ,det))

        return x


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: int = 0
    norm_layer: nn.Module = None  # Optional normalization layer

    @nn.compact
    def __call__(self, x):
        # Apply normalization if provided
        if self.norm_layer:
            x = self.norm_layer(name='norm',dtype=dtype)(x)

        # Convolution operation
        x = Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            use_bias=True,  # Flax uses `use_bias` instead of `bias` keyword
            name="conv",dtype = dtype
        )(x)
        return x


class MetaFormerStage(nn.Module):
    in_chs: int
    out_chs: int
    depth: int = 2
    token_mixer: Callable[..., nn.Module] = SepConv
    mlp_act: Callable[..., nn.Module] = StarReLU
    mlp_bias: bool = False
    downsample_norm: nn.Module = partial(nn.LayerNorm, use_fast_variance=use_fast_variance)
    norm_layer: nn.Module = partial(nn.LayerNorm, use_fast_variance=use_fast_variance)
    proj_drop: float = 0.0
    dp_rates: list = field(default_factory=list)
    layer_scale_init_value: float = None
    res_scale_init_value: float = None
    grad_checkpointing: bool = False
    use_nchw: bool = True

    @nn.compact
    def __call__(self, x,det=True):
        # Downsampling layer (identity if in_chs == out_chs)
        downsample = Identity() if self.in_chs == self.out_chs else Downsampling(
            self.out_chs, kernel_size=(3, 3), stride=2, padding=1, norm_layer=self.downsample_norm,name='downsample')

        x = downsample(x)

        B, H, W,C = x.shape
        use_nchw = True
        if  issubclass(self.token_mixer,Attention):
            use_nchw=False
            x=einops.rearrange(x,'b  h w c-> b (h w) c')

        # Create MetaFormerBlocks
        for i in range(self.depth):
            block = MetaFormerBlock(
                dim=self.out_chs,
                token_mixer=self.token_mixer,
                mlp_act=self.mlp_act,
                mlp_bias=self.mlp_bias,
                norm_layer=self.norm_layer,
                proj_drop=self.proj_drop,
                drop_path=self.dp_rates[i],
                layer_scale_init_value=self.layer_scale_init_value,
                res_scale_init_value=self.res_scale_init_value,
                use_nchw=use_nchw,
            )
            x = block(x,det)

        if  issubclass(self.token_mixer,Attention):
            x=einops.rearrange(x,'b (h w) c->  b  h w c',h=H,w=W)

        return x



class Silu(nn.Module):

    @nn.compact
    def __call__(self, x):
        return nn.silu(x)


class MlpHead(nn.Module):
    dim: int
    num_classes: int = 1000
    mlp_ratio: float = 4.0
    act_layer: nn.Module = SquaredReLU
    norm_layer: nn.Module = nn.LayerNorm
    head_dropout: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x,det=True):
        hidden_features = int(self.mlp_ratio * self.dim)

        # First fully connected layer
        x = Dense(features=hidden_features, use_bias=self.bias,name='fc1')(x)
        x = self.act_layer()(x)

        # Normalization
        x = self.norm_layer(name='norm')(x)
        # Dropout
        x = nn.Dropout(rate=self.head_dropout, deterministic=det)(x)

        # Second fully connected layer
        x = Dense(features=self.num_classes, use_bias=self.bias,name='fc2')(x)

        return x


class MetaFormer(nn.Module):
    """
    MetaFormer - A Flax implementation inspired by PyTorch's MetaFormer.

    Args:
        in_chans (int): Number of input channels.
        num_classes (int): Number of output classes for classification.
        global_pool (str): Type of global pooling to use for the classification head.
        depths (Sequence[int]): Number of blocks at each stage.
        dims (Sequence[int]): Feature dimensions at each stage.
        token_mixers (Union[nn.Module, Sequence[nn.Module]]): Token mixers for each stage.
        mlp_act (Callable): Activation function for MLP layers.
        mlp_bias (bool): Whether to include bias in MLP layers.
        drop_path_rate (float): Rate for stochastic depth.
        proj_drop_rate (float): Dropout rate for projection layers.
        drop_rate (float): Dropout rate for general use.
        layer_scale_init_values (Optional[Sequence[Optional[float]]]): Initial values for layer scaling.
        res_scale_init_values (Sequence[Optional[float]]): Initial values for residual connection scaling.
        downsample_norm (Callable): Normalization function for downsampling.
        norm_layers (Union[Callable, Sequence[Callable]]): Normalization function(s) for stages.
        output_norm (Callable): Normalization function for the output before the classification head.
        use_mlp_head (bool): Whether to use MLP head.
    """
    in_chans: int = 3
    num_classes: int = 1000
    labels: int = 1000
    global_pool: str = 'avg'
    depths: Sequence[int] = (2, 2, 6, 2)
    dims: Sequence[int] = (64, 128, 320, 512)
    token_mixers: Union[nn.Module, Sequence[nn.Module]] = SepConv
    mlp_act: Callable = StarReLU
    mlp_bias: bool = False
    drop_path_rate: float = 0.0
    proj_drop_rate: float = 0.0
    drop_rate: float = 0.0
    layer_scale_init_values: Optional[Sequence[Optional[float]]] = None
    res_scale_init_values: Sequence[Optional[float]] = (None, None, 1.0, 1.0)
    downsample_norm: Callable = partial(nn.LayerNorm,use_fast_variance=use_fast_variance,use_bias=False)
    norm_layers: Union[Callable, Sequence[Callable]] = partial(nn.LayerNorm,use_fast_variance=use_fast_variance,use_bias=False)
    output_norm: Callable = nn.LayerNorm
    use_mlp_head: bool = True
    mlp_head_act: Any =SquaredReLU

    @nn.compact
    def __call__(self, x,det=True):
        x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        # Convert input parameters to appropriate format if needed
        depths = list(self.depths)
        dims = list(self.dims)
        token_mixers = [self.token_mixers] * len(depths) if not isinstance(self.token_mixers, (list, tuple)) else self.token_mixers
        norm_layers = [self.norm_layers] * len(depths) if not isinstance(self.norm_layers, (list, tuple)) else self.norm_layers
        layer_scale_init_values = [self.layer_scale_init_values] * len(depths) if self.layer_scale_init_values is not None else [None] * len(depths)
        res_scale_init_values = self.res_scale_init_values

        # Stem layer
        x = Stem( dims[0], norm_layer=self.downsample_norm,name='stem')(x)

        # Stages
        dp_rates = np.linspace(0, self.drop_path_rate, sum(depths))
        dp_rates = np.split(dp_rates, np.cumsum(depths)[:-1])  # Split according to depths

        # Convert each segment to a list if needed
        dp_rates = [segment.tolist() for segment in dp_rates]
        prev_dim = dims[0]
        for i in range(len(depths)):
            stage = MetaFormerStage(
                prev_dim,
                    dims[i],
                    token_mixer=token_mixers[i],
                    mlp_act=self.mlp_act,
                    mlp_bias=self.mlp_bias,
                    proj_drop=self.proj_drop_rate,
                    dp_rates=dp_rates[i],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i],
                    downsample_norm=self.downsample_norm,
                    norm_layer=norm_layers[i],
                    depth=self.depths[i]
                )

            prev_dim = dims[i]
            x = stage(x,det)

        # Output normalization and head
        x = self.output_norm(name='out_norm')(x.mean(axis=(1, 2)))  # Global pooling, assuming (B, H, W, C)
        if self.num_classes > 0:
            if self.use_mlp_head:
                if self.mlp_head_act=='silu':
                    mlp_head_act=Silu
                else:
                    mlp_head_act=self.mlp_head_act

                x = MlpHead(dims[-1],self.num_classes,name='fc',head_dropout=self.drop_rate,act_layer=mlp_head_act)(x,det)
            else:
                x = Dense(self.num_classes)(x)
        return x



ReMatSepConv=nn.remat(SepConv)
ReMatAttention=nn.remat(Attention)



CAFormer=partial(MetaFormer,token_mixers=(SepConv,SepConv,Attention,Attention))
# CAFormer=partial(MetaFormer,token_mixers=(ReMatSepConv,ReMatSepConv,ReMatAttention,ReMatAttention))
ConvFormer=partial(MetaFormer,token_mixers=(SepConv,SepConv,SepConv,SepConv))