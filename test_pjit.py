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
from copy import deepcopy

# os.environ['GOPEN_VERBOSE'] = '1'

import jax
# jax.distributed.initialize()

import time

import einops
import flax.jax_utils
import orbax.checkpoint
import orbax.checkpoint as ocp

import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.serialization import msgpack_serialize
from flax.training import orbax_utils
from flax.training.common_utils import shard
from jax import NamedSharding
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax._src.pjit import pjit
from jax.experimental import multihost_utils
from tensorboard.plugins.image.summary import image
from torch.nn.parallel import replicate
from torch.utils.data import DataLoader

from pre_define import TRAIN_EVAL_FN_COLLECTION
from state.state_pjit2 import init_state
from test_dataset_fork2 import create_dataloaders, DynamicMixRatioState
# from test_dataset_fork import create_dataloaders
from jax.sharding import PartitionSpec as P
from training_pjit import TrainState, training_step, validation_adv_step, training_step_kl
from utils import AverageMeter, read_yaml, preprocess_config, save_checkpoint_in_background, \
    save_checkpoint_in_background2, match_partition_rules, get_jax_mesh2

import jax.tree_util as jtu
from functools import partial
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning


def _build_global_shape_and_sharding(
        local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
    sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

    return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
    """Put local sharded array into local devices"""

    global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
    try:
        local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
    except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(global_mesh.local_devices)} "
            f"at {jtu.keystr(path)}"
        ) from array_split_error

    local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
    return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


def evaluate(state: TrainState, dataloader: DataLoader, validation_adv_step_jited, mesh) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        batch = jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(x)), batch)
        batch = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), batch)
        metrics = validation_adv_step_jited(state, batch)

        print(metrics)

        average_meter.update(**metrics)

    metrics = average_meter.summary("val/")
    num_samples = metrics.pop("val/num_samples")
    return jax.tree_util.tree_map(lambda x: x / num_samples, metrics)


def main(configs):
    training_steps = configs['steps'] * configs['training_epoch'] // configs['dataset']['train_batch_size']
    warmup_steps = configs['steps'] * configs['warmup_epoch'] // configs['dataset']['train_batch_size']
    eval_interval = configs['steps'] * configs['eval_epoch'] // configs['dataset']['train_batch_size']
    epoch_per_step = configs['steps'] // configs['dataset']['train_batch_size']
    log_interval = configs['log_interval']
    use_orbax_save = configs.pop('use_orbax_save', True)
    valid_fn = configs.pop('valid_fn', "validation_adv_step")
    train_fn = configs.pop('train_fn', "training_step")
    grad_accum_steps = configs['train_state'].get('grad_accum_steps', 1)
    resume = configs.get('resume', False)

    # os.environ['JAX_PLATFORMS']='cpu'
    # os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
    # jax.config.update('jax_platform_name', 'cpu')
    # pass

    use_pgd = configs.pop('use_pgd', True)

    dp = configs.pop('dp', -1)
    fsdp = configs.pop('fsdp', 1)
    tp = configs.pop('tp', 1)

    mesh_dim = f'{dp},{fsdp},{tp}'  #  '-1,1,1'

    postfix = "ema"
    name = configs['name']
    output_dir = configs['output_dir']
    filename = os.path.join(output_dir, f"{name}-{postfix}")
    print(filename)

    # mesh_dim = '1,1,-1'
    # mesh_dim = '1,1,-1'
    # mesh_dim = '1,-1,1'


    train_dataloader_iter, valid_dataloader, mix_ratio_state = create_dataloaders(**configs['dataset'],
                                                                                  grad_accum=grad_accum_steps)



    # for _ in valid_dataloader:
    #     break



    mesh = get_jax_mesh2(mesh_dim)
    # print(mesh)
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("dp", 'fsdp', 'mp'))
    # print(sharding)
    data_spec = [["dp", 'fsdp', 'mp']]
    # data_spec=["dp",'fsdp','mp']
    data_spec = P(*data_spec)
    print(data_spec)
    sharding = jtu.tree_map(lambda p: NamedSharding(mesh, p), data_spec)
    # print(sharding.addressable_devices,mesh.axis_names)



    logical_axis_rules = [
        ['batch', ['dp', 'fsdp']],
        ['activation_embed', 'mp'],
        ['mlp', 'mp'],
        ['vocab', 'fsdp'],
        ['embed', 'fsdp'],
        ['heads', 'mp'],
    ]

    with mesh, nn_partitioning.axis_rules(logical_axis_rules):
        valid_step = TRAIN_EVAL_FN_COLLECTION[valid_fn]
        train_step = TRAIN_EVAL_FN_COLLECTION[train_fn]

        state, init_step, train_state_sharding = init_state(configs['train_state'],
                                                            warmup_steps=warmup_steps,
                                                            training_steps=training_steps, mesh=mesh,
                                                            restore_state_config=configs[
                                                                'restore_state'] if 'restore_state' in configs else None,
                                                            remote_model_path=filename, resume=resume)

        training_step_pjit = jax.jit(train_step, static_argnums=(2,),
                                     donate_argnums=(0,),
                                     out_shardings=(train_state_sharding, None),
                                     # in_shardings=(train_state_sharding, sharding,),
                                     )

        opt_state = jax.tree_util.tree_map(
            lambda x: x.with_memory_kind(kind="pinned_host"), train_state_sharding.opt_state)

        params = jax.tree_util.tree_map(
            lambda x: x.with_memory_kind(kind="pinned_host"), train_state_sharding.params)
        train_state_off_load_sharding = train_state_sharding.replace(params=params,
                                                                     opt_state=opt_state)

        def change_state_device(state):
            return state


        off_load_memory_state=jax.jit(change_state_device, out_shardings=train_state_off_load_sharding)
        reload_device_state = jax.jit(change_state_device, out_shardings=train_state_sharding)





        validation_adv_step_jited = jax.jit(valid_step,
                                            # in_shardings=(
                                            #     train_state_off_load_sharding, NamedSharding(mesh, P(('dp', 'fsdp', 'mp')))),
                                            # donate_argnums=(0,),
                                            out_shardings=None
                                            )


        checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
        # checkpointer =ocp.PyTreeCheckpointer()

        average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0

        epoch = init_step // epoch_per_step
        mix_ratio_state.update_mix_ratio(epoch, configs['training_epoch'])

        # if jax.process_index() == 0:
        #     wandb.init(name=configs['name'], project=configs['project'], config=configs)

        metrics = evaluate(state, valid_dataloader, validation_adv_step_jited, mesh)

        if "val/advacc1" in metrics:
            now_acc1 = metrics["val/advacc1"]
        else:
            now_acc1 = metrics["val/acc1"]
        print(now_acc1,max_val_acc1)
        print()


            # metrics["val/acc1/best"] = max_val_acc1
            # metrics["processed_samples"] = step * configs['dataset']['train_batch_size']
            # if jax.process_index() == 0:
            #     wandb.log(metrics, step)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", type=str,
                        default='configs/scaling/eval/convnext-b-224-300ep-rar-50m.yaml')

    args = parser.parse_args()
    yaml = read_yaml(args.yaml_path)
    yaml = preprocess_config(yaml)

    try:
        jax.distributed.initialize()
        main(yaml)
    except Exception as e:
        print(e)
