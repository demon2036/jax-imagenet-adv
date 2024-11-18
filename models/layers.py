from functools import partial
from typing import Optional, Callable
import flax.linen as nn
from torch.backends.cudnn import deterministic


class Mlp(nn.Module):
    """
    Flax implementation of the MLP as used in Vision Transformer, MLP-Mixer, etc.
    """
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable[..., nn.Module] = nn.gelu
    norm_layer: Optional[Callable[..., nn.Module]] = None
    bias: bool = True
    drop: float = 0.0
    use_conv: bool = False

    @nn.compact
    def __call__(self, x,det=True):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features

        # Choose the linear or convolutional layer
        linear_layer = partial(nn.Conv, kernel_size=(1, 1)) if self.use_conv else nn.Dense

        # First layer
        x = linear_layer(features=hidden_features, use_bias=self.bias,name='fc1')(x)
        x = self.act_layer(name='act')(x)
        # x = self.act_layer(x,approximate=False)
        x = nn.Dropout(self.drop)(x,deterministic=det)

        # Optional normalization
        if self.norm_layer:
            x = self.norm_layer()(x)

        # Second layer
        x = linear_layer(features=out_features, use_bias=self.bias,name='fc2')(x)
        x = nn.Dropout(self.drop)(x,deterministic=det)

        return x


class DropPath(nn.Module):
    drop_path:float =0.0

    def __call__(self, x,det=True):
        x=nn.Dropout(self.droppath, broadcast_dims=x.shape[1:])(x,deterministic=det)
        return x