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

import einops
import flax.jax_utils
import orbax.checkpoint as ocp
import jax
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
from tensorboard.plugins.image.summary import image
from torch.nn.parallel import replicate
from torch.utils.data import DataLoader

from state import create_train_state
from test_dataset_fork2 import create_dataloaders, DynamicMixRatioState
# from test_dataset_fork import create_dataloaders
from jax.sharding import PartitionSpec as P
from training_pjit import TrainState, training_step, validation_adv_step
from utils import AverageMeter, read_yaml, preprocess_config, save_checkpoint_in_background, \
    save_checkpoint_in_background2, get_partition_rules, match_partition_rules, get_jax_mesh2

import jax.tree_util as jtu
from functools import partial
import jax.numpy as jnp

# warnings.filterwarnings("ignore")
# os.environ['LIBTPU_INIT_ARGS']='--xla_enable_async_all_gather=auto '
# os.environ['XLA_FLAGS'] = '--xla_enable_async_all_gather=auto '
# os.environ['XLA_FLAGS'] = '--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_enable_async_all_gather=true   --xla_tpu_enable_async_collective_fusion=true --xla_tpu_megacore_fusion_allow_ags=true'


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
    epoch_per_step = configs['steps'] // configs['dataset']['train_batch_size']
    log_interval = configs['log_interval']
    use_orbax_save = configs.pop('use_orbax_save', True)


    if use_orbax_save:
        # os.environ['JAX_PLATFORMS']='cpu'
        # os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
        # jax.config.update('jax_platform_name', 'cpu')
        # pass
        jax.distributed.initialize()

    use_pgd = configs.pop('use_pgd', True)
    grad_accum_steps = configs.pop('grad_accum_steps', 1)

    # if jax.process_index() == 0:
    #     pass
        # wandb.init(name=configs['name'], project=configs['project'], config=configs)

    postfix = "ema"
    name = configs['name']
    output_dir = configs['output_dir']
    filename = os.path.join(output_dir, f"{name}-{postfix}")
    print(filename)

    mesh_dim = '-1,1,4'
    mesh = get_jax_mesh2(mesh_dim)
    print(mesh)
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("dp",'fsdp','mp'))
    print(sharding)
    data_spec=[["dp",'fsdp','mp']]
    # data_spec=["dp",'fsdp','mp']
    data_spec=P(*data_spec)
    print(data_spec)
    sharding=jtu.tree_map(lambda p:NamedSharding(mesh,p),data_spec)
    print(sharding.addressable_devices,mesh.axis_names)
    """
    num_model_replicas_per_process = 4  # set according to your parallelism strategy
    num_model_replicas_total = num_model_replicas_per_process * jax.process_count()

    # Create an example `Mesh` for per-process data parallelism. Make sure all devices
    # are grouped by process, and then resize so each row is a model replica.
    mesh_devices = np.array([jax.local_devices(process_idx)
                             for process_idx in range(jax.process_count())])

    print(mesh_devices.shape)
    mesh= mesh_devices.reshape(4,1, -1)
    mesh = einops.rearrange(mesh, 'a b c -> c b a')
    mesh=Mesh(mesh, ('dp', 'fsdp', 'mp'))

    mesh_devices = mesh_devices.reshape(num_model_replicas_total,1, -1)
    mesh_devices=einops.rearrange(mesh_devices,'a b c -> c b a')
    print(mesh_devices.shape)
    mesh_data = jax.sharding.Mesh(mesh_devices, ['dp','fsdp','mp'])

    # Shard the data across model replicas. You don't shard across the
    # data_parallelism mesh axis, meaning each per-replica shard will be replicated
    # across that axis.
    sharding = jax.sharding.NamedSharding(
        mesh_data, jax.sharding.PartitionSpec("mp"))


    # while True:
    #     pass
    """
    # x=jnp.ones((128,3,224,224))
    # batch = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), x)
    # jax.debug.visualize_array_sharding(batch[:,:,0,0,])
    # print('\n'*5)
    #
    # @partial(jax.jit,out_shardings=sharding)
    # def test(x):
    #     return x
    #
    # jax.debug.visualize_array_sharding(test(batch[:,:,0,0,]))
    # print(1)
    # while True:
    #     pass


    train_dataloader_iter, valid_dataloader, mix_ratio_state = create_dataloaders(**configs['dataset'],
                                                                                  grad_accum=grad_accum_steps)
    with mesh:

        state, train_state_partition = create_train_state(configs['train_state'],
                                                          warmup_steps=warmup_steps,
                                                          training_steps=training_steps,
                                                          grad_accum_steps=grad_accum_steps, mesh=mesh)

        train_state_sharding = jtu.tree_map(lambda x: NamedSharding(mesh, x), train_state_partition)
        # state=state.replace(step=1)



        def go(x):
            return x

        go_jit=jax.jit(go,
                       # out_shardings=sharding
                       out_shardings=jax.NamedSharding(mesh,P('dp'))
                       )


        training_step_pjit = jax.jit(training_step, static_argnums=(2,),
                                     donate_argnums=(0,),
                                     # in_shardings=(train_state_sharding, sharding,),
                                     out_shardings=(train_state_sharding,None ))

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

        average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0

        epoch = init_step // epoch_per_step
        mix_ratio_state.update_mix_ratio(epoch, configs['training_epoch'])
        for step in tqdm.tqdm(range(init_step, training_steps + 1), initial=init_step, total=training_steps + 1):
            # for step in tqdm.trange(init_step, training_steps + 1, dynamic_ncols=True):
            for _ in range(grad_accum_steps):
                # batch = jax.tree_util.tree_map(lambda x: jax.make_array_from_process_local_data(sharding,np.asarray(x))  , next(train_dataloader_iter))
                batch = jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(x)), next(train_dataloader_iter))
                batch = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), batch)

                batch = jtu.tree_map(go_jit, batch)


                images, labels = batch
                print(f'{images.shape=}   {images.addressable_data(0).shape=}')

                jax.debug.visualize_array_sharding(labels,max_width=200)
                print()

                di=flax.traverse_util.flatten_dict(state.params,sep='.')
                print(di.keys())
                jax.debug.visualize_array_sharding(di['model.MetaFormerStage_2.MetaFormerBlock_7.mlp.fc1.kernel'])
                while True:
                    pass

                state, metrics = training_step_pjit(state, batch, use_pgd)
                # images,labels=batch

                # print(f'{images.shape=}  {labels.shape=}')
                # if jax.process_index() == 0:
                #     images, labels = batch
                #     print(f'{images.shape=}   {images.addressable_data(0).shape=}')
                #     images, labels = next(train_dataloader_iter)
                #     print(f'{images.shape=}')


                images, labels = next(train_dataloader_iter)
                print(f'{images.shape=}')

                # print(metrics)

                # while True:
                #     pass

                # state, metrics = training_step(state, batch, use_pgd)
                # average_meter.update(**unreplicate(metrics))

        # print(state.params)

    # train_state_shapes = jax.eval_shape(init_fn, params)
    # train_state_partition = match_partition_rules(get_partition_rules(), train_state_shapes)

    """







    epoch = init_step // epoch_per_step
    mix_ratio_state.update_mix_ratio(epoch, configs['training_epoch'])
    for step in tqdm.tqdm(range(init_step, training_steps + 1), initial=init_step, total=training_steps + 1):
        # for step in tqdm.trange(init_step, training_steps + 1, dynamic_ncols=True):
        for _ in range(grad_accum_steps):
            batch = shard(jax.tree_util.tree_map(np.asarray, next(train_dataloader_iter)))
            state, metrics = training_step(state, batch, use_pgd)
            average_meter.update(**unreplicate(metrics))


        if step%epoch_per_step==0:
            epoch=step//epoch_per_step
            mix_ratio_state.update_mix_ratio(epoch,configs['training_epoch'])

        if (
                jax.process_index() == 0
                and log_interval > 0
                and step % log_interval == 0
        ):
            metrics = average_meter.summary(prefix="train/")
            metrics["processed_samples"] = step * configs['dataset']['train_batch_size']
            metrics["mix_ratio"] = mix_ratio_state.ratio
            wandb.log(metrics, step)

        if eval_interval > 0 and (
                step % eval_interval == 0 or step == training_steps
        ):
            if valid_dataloader is None:
                continue
            try:
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
            except Exception as e:
                print(e)
    if use_orbax_save:
        checkpointer.wait_until_finished()
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", type=str,
                        default='configs/planB/ablation/best/test.yaml')
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
    args = parser.parse_args()
    yaml = read_yaml(args.yaml_path)
    # yaml = read_yaml('configs/adv/convnext-b-3step.yaml')
    # yaml = read_yaml('configs/adv/convnext-t-3step.yaml')
    yaml = preprocess_config(yaml)

    # print(yaml)
    # while True:
    #     pass

    main(yaml)
