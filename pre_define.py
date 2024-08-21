from __future__ import annotations

import optax

from utils import modified_lamb
import numpy as np

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}
OPTIMIZER_COLLECTION = {
    "adamw": optax.adamw,
    "lamb": modified_lamb,
    'lion': optax.lion
}