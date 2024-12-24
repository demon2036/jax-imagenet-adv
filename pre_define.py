from __future__ import annotations

import jax.numpy
import optax

from optimizer.modified_lion import modified_lion
from optimizer.muon import muon
from training_pjit import validation_step, validation_adv_step
from utils import modified_lamb
import numpy as np

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])



def softmax_cross_entropy_with_z_loss(logits,labels,z_loss_weight=2e-4):
    ce_loss=optax.softmax_cross_entropy(logits,labels)
    softmax_normalizer = logits.max(-1) ** 2
    z_loss = z_loss_weight * softmax_normalizer.mean()
    return ce_loss+z_loss



CRITERION_COLLECTION = {
    "ce_z_loss":softmax_cross_entropy_with_z_loss,
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}
OPTIMIZER_COLLECTION = {
    "adamw": optax.adamw,
    "lamb": modified_lamb,
    'muon': muon,
    "modified_lion": modified_lion,
    # "lamb": optax.lamb,
    'lion': optax.lion,
    'sgd':optax.sgd
}



TRAIN_EVAL_FN_COLLECTION = {
    'validation_step':validation_step,
    'validation_adv_step':validation_adv_step,
}

