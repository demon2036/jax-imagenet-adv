# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import argparse
import functools
import random
import warnings

import einops
import jax
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.serialization import msgpack_serialize
from flax.training.common_utils import shard, shard_prng_key
from optax import softmax_cross_entropy_with_integer_labels
from torch.utils.data import DataLoader
from robust.training import TrainState, create_train_state, training_step, validation_step
from utils import AverageMeter, save_checkpoint_in_background

# from dataset_mix import create_dataloaders
from dataset import create_dataloaders
from attacks.pgd import pgd_attack


# warnings.filterwarnings("ignore")


def evaluate(state: TrainState, dataloader: DataLoader) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        metrics = validation_step(state, shard(jax.tree_map(np.asarray, batch)))
        average_meter.update(**jax.device_get(unreplicate(metrics)))

    metrics = average_meter.summary("val/")
    num_samples = metrics.pop("val/num_samples")
    return jax.tree_util.tree_map(lambda x: x / num_samples, metrics)


def main(args: argparse.Namespace):
    train_dataloader, valid_dataloader = create_dataloaders(args)
    train_dataloader_iter = iter(train_dataloader)
    # train_dataloader_iter = train_dataloader
    state = create_train_state(args).replicate()

    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args)
    average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0

    key = jax.random.PRNGKey(1)
    key = shard_prng_key(key)
    import jax.numpy as jnp
    batch = shard(jax.tree_util.tree_map(np.asarray, next(train_dataloader_iter)))
    img = batch[0]

    @functools.partial(jax.pmap)
    def test(images, label, state, key):
        images = einops.rearrange(images, 'b c h w->b h w c')
        images = images.astype(jnp.float32)

        # label = label.astype(jnp.int32)

        # image_perturbation = jnp.zeros_like(image)
        image_perturbation = jax.random.uniform(key, images.shape, minval=-4, maxval=4)
        return state.apply_fn({"params": state.params}, images + image_perturbation)
        # return state.apply_fn({'params': state.params}, images)

    def adversarial_loss(perturbation, state, image, label):
        logits = state.apply_fn({"params": state.params}, image + perturbation)
        loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, label))
        # loss_value = logits
        return loss_value

    @functools.partial(jax.pmap)
    def test2(image, label, state, epsilon=4 / 255, step_size=4 / 3 / 255, maxiter=1, key=None):

        image = einops.rearrange(image, 'b c h w->b h w c')
        image = image.astype(jnp.float32)

        label = label.astype(jnp.int32)

        # image_perturbation = jnp.zeros_like(image)
        image_perturbation = jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)

        grad_adversarial = jax.grad(adversarial_loss)
        return grad_adversarial(image_perturbation, state, image, label)
        for _ in range(maxiter):
            # compute gradient of the loss wrt to the image
            sign_grad = jnp.sign(grad_adversarial(image_perturbation, state, image, label))

            # heuristic step-size 2 eps / maxiter
            image_perturbation += step_size * sign_grad
            # projection step onto the L-infinity ball centered at image
            image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)



        # clip the image to ensure pixels are between 0 and 1
        return jnp.clip(image + image_perturbation, 0, 1)

    # pgd_attack_pmap = jax.pmap(pgd_attack)

    for step in tqdm.trange(1, args.training_steps + 1, dynamic_ncols=True):
        # state, metrics = training_step(state, batch)
        # out = pgd_attack_pmap(batch[0], batch[1], state, key=key)
        out = test2(batch[0], batch[1], state, key=key)
        # pgd_attack(b)

    for step in tqdm.trange(1, args.training_steps + 1, dynamic_ncols=True):
        for _ in range(args.grad_accum):
            batch = shard(jax.tree_util.tree_map(np.asarray, next(train_dataloader_iter)))
            state, metrics = training_step(state, batch)
            average_meter.update(**unreplicate(metrics))

        if (
                jax.process_index() == 0
                and args.log_interval > 0
                and step % args.log_interval == 0
        ):
            metrics = average_meter.summary(prefix="train/")
            metrics["processed_samples"] = step * args.train_batch_size
            wandb.log(metrics, step)

        if args.eval_interval > 0 and (
                step % args.eval_interval == 0 or step == args.training_steps
        ):
            if jax.process_index() == 0:
                params_bytes = msgpack_serialize(unreplicate(state.params))
                save_checkpoint_in_background(args, params_bytes, postfix="last")
            if valid_dataloader is None:
                continue

            metrics = evaluate(state, valid_dataloader)
            if jax.process_index() == 0:
                if metrics["val/acc1"] > max_val_acc1:
                    max_val_acc1 = metrics["val/acc1"]
                    save_checkpoint_in_background(args, params_bytes, postfix="best")

                metrics["val/acc1/best"] = max_val_acc1
                metrics["processed_samples"] = step * args.train_batch_size
                wandb.log(metrics, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--valid-dataset-shards")
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--valid-batch-size", type=int, default=256)
    parser.add_argument("--train-loader-workers", type=int, default=40)
    parser.add_argument("--valid-loader-workers", type=int, default=5)

    parser.add_argument("--random-crop", default="rrc")
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--random-erasing", type=float, default=0.25)
    parser.add_argument("--augment-repeats", type=int, default=3)
    parser.add_argument("--test-crop-ratio", type=float, default=0.875)

    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--criterion", default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--labels", type=int, default=-1)
    parser.add_argument("--layerscale", action="store_true", default=False)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--posemb", default="learnable")
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--droppath", type=float, default=0.1)
    parser.add_argument("--grad-ckpt", action="store_true", default=False)

    parser.add_argument("--init-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--mixup-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--dropout-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--shuffle-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--pretrained-ckpt")
    parser.add_argument("--label-mapping")

    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)

    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--training-steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=0)

    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default=".")
    jax.distributed.initialize()
    main(parser.parse_args())
