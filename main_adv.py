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
import os
import time

import orbax.checkpoint as ocp
import jax
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.serialization import msgpack_serialize
from flax.training import orbax_utils
from flax.training.common_utils import shard
from torch.utils.data import DataLoader

from test_dataset_fork import create_dataloaders
from test_state import create_train_state
from training import TrainState, training_step, validation_adv_step
from utils import AverageMeter, read_yaml, preprocess_config, save_checkpoint_in_background, \
    save_checkpoint_in_background2


# from dataset import create_dataloaders


# warnings.filterwarnings("ignore")


def evaluate(state: TrainState, dataloader: DataLoader) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        metrics = validation_adv_step(state, shard(jax.tree_map(np.asarray, batch)))
        average_meter.update(**jax.device_get(unreplicate(metrics)))

    metrics = average_meter.summary("val/")
    num_samples = metrics.pop("val/num_samples")
    return jax.tree_util.tree_map(lambda x: x / num_samples, metrics)


def main(configs):

    training_steps = configs['steps'] * configs['training_epoch'] // configs['dataset']['train_batch_size']
    warmup_steps = configs['steps'] * configs['warmup_epoch'] // configs['dataset']['train_batch_size']
    eval_interval = configs['steps'] * configs['eval_epoch'] // configs['dataset']['train_batch_size']
    log_interval = configs['log_interval']

    use_orbax_save=configs.pop('use_orbax_save',True)
    if use_orbax_save:
        jax.distributed.initialize()


    if jax.process_index() == 0:
        wandb.init(name=configs['name'], project=configs['project'], config=configs)




    state = create_train_state(configs['train_state'], warmup_steps=warmup_steps,
                               training_steps=training_steps)


    postfix = "ema"
    name = configs['name']
    output_dir = configs['output_dir']
    filename = os.path.join(output_dir, f"{name}-{postfix}")
    print(filename)


    if use_orbax_save:
        checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
        ckpt = {'model': state}

        if 'resume' in configs:
            state = checkpointer.restore(filename, item=ckpt)['model']
            init_step = state.step + 1
            del ckpt
        else:
            init_step = 1
    else:
        init_step = 1

    state = state.replicate()

    train_dataloader, valid_dataloader = create_dataloaders(**configs['dataset'])
    # train_dataloader_iter = iter(train_dataloader)
    train_dataloader_iter = train_dataloader
    average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0
    for step in tqdm.tqdm(range(init_step, training_steps + 1), initial=init_step, total=training_steps + 1):
    # for step in tqdm.trange(init_step, training_steps + 1, dynamic_ncols=True):
        for _ in range(1):
            batch = shard(jax.tree_util.tree_map(np.asarray, next(train_dataloader_iter)))
            state, metrics = training_step(state, batch)
            average_meter.update(**unreplicate(metrics))

        if (
                jax.process_index() == 0
                and log_interval > 0
                and step % log_interval == 0
        ):
            metrics = average_meter.summary(prefix="train/")
            metrics["processed_samples"] = step * configs['dataset']['train_batch_size']
            wandb.log(metrics, step)

        if eval_interval > 0 and (
                step % eval_interval == 0 or step == training_steps
        ):
            if valid_dataloader is None:
                continue

            metrics = evaluate(state, valid_dataloader)

            if metrics["val/advacc1"] > max_val_acc1:
                if use_orbax_save:
                    ckpt = {'model': jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))}
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    checkpointer.save(filename, ckpt, save_args=save_args, force=True)
                else:
                    if jax.process_index() == 0:
                        params_bytes = msgpack_serialize(unreplicate(state.ema_params))
                        save_checkpoint_in_background(filename, params_bytes, postfix="last")

                max_val_acc1 = metrics["val/advacc1"]
                # save_checkpoint_in_background(args, params_bytes, postfix="best")

            metrics["val/acc1/best"] = max_val_acc1
            metrics["processed_samples"] = step * configs['dataset']['train_batch_size']
            if jax.process_index() == 0:
                wandb.log(metrics, step)

    checkpointer.wait_until_finished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", type=str, default='configs/ablation/amount_data/conv-next-b-224-3step-300ep-mix0.9-adv-step-3-1m.yaml')
    # parser.add_argument("--train-dataset-shards")
    # parser.add_argument("--valid-dataset-shards")
    # parser.add_argument("--train-batch-size", type=int, default=2048)
    # parser.add_argument("--valid-batch-size", type=int, default=256)
    # parser.add_argument("--train-loader-workers", type=int, default=40)
    # parser.add_argument("--valid-loader-workers", type=int, default=5)
    #
    # parser.add_argument("--random-crop", default="rrc")
    # parser.add_argument("--color-jitter", type=float, default=0.0)
    # parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    # parser.add_argument("--random-erasing", type=float, default=0.25)
    # parser.add_argument("--augment-repeats", type=int, default=3)
    # parser.add_argument("--test-crop-ratio", type=float, default=0.875)
    #
    # parser.add_argument("--mixup", type=float, default=0.8)
    # parser.add_argument("--cutmix", type=float, default=1.0)
    # parser.add_argument("--criterion", default="ce")
    # parser.add_argument("--label-smoothing", type=float, default=0.1)
    #
    # parser.add_argument("--layers", type=int, default=12)
    # parser.add_argument("--dim", type=int, default=768)
    # parser.add_argument("--heads", type=int, default=12)
    # parser.add_argument("--labels", type=int, default=-1)
    # parser.add_argument("--layerscale", action="store_true", default=False)
    # parser.add_argument("--patch-size", type=int, default=16)
    # parser.add_argument("--image-size", type=int, default=224)
    # parser.add_argument("--posemb", default="learnable")
    # parser.add_argument("--pooling", default="cls")
    # parser.add_argument("--dropout", type=float, default=0.0)
    # parser.add_argument("--droppath", type=float, default=0.1)
    # parser.add_argument("--grad-ckpt", action="store_true", default=False)
    #
    # parser.add_argument("--init-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--mixup-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--dropout-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--shuffle-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--pretrained-ckpt")
    # parser.add_argument("--label-mapping")
    #
    # parser.add_argument("--optimizer", default="adamw")
    # parser.add_argument("--learning-rate", type=float, default=1e-3)
    # parser.add_argument("--weight-decay", type=float, default=0.05)
    # parser.add_argument("--adam-b1", type=float, default=0.9)
    # parser.add_argument("--adam-b2", type=float, default=0.999)
    # parser.add_argument("--adam-eps", type=float, default=1e-8)
    # parser.add_argument("--lr-decay", type=float, default=1.0)
    # parser.add_argument("--clip-grad", type=float, default=0.0)
    # parser.add_argument("--grad-accum", type=int, default=1)
    #
    # parser.add_argument("--warmup-steps", type=int, default=10000)
    # parser.add_argument("--training-steps", type=int, default=200000)
    # parser.add_argument("--log-interval", type=int, default=50)
    # parser.add_argument("--eval-interval", type=int, default=0)
    #
    # parser.add_argument("--project")
    # parser.add_argument("--name")
    # parser.add_argument("--ipaddr")
    # parser.add_argument("--hostname")
    # parser.add_argument("--output-dir", default=".")
    # main(parser.parse_args())
    args=parser.parse_args()
    yaml = read_yaml(args.yaml_path)
    # yaml = read_yaml('configs/adv/convnext-b-3step.yaml')
    # yaml = read_yaml('configs/adv/convnext-t-3step.yaml')
    yaml = preprocess_config(yaml)

    # print(yaml)
    # while True:
    #     pass




    main(yaml)
