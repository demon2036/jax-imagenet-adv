import copy
import json
import time
from functools import partial

import flax
import jax
import numpy as np
import optax
import timm
from jax._src.pjit import pjit

import train_modules
from pre_define import CRITERION_COLLECTION, OPTIMIZER_COLLECTION
from training import TrainState
from utils import read_yaml, get_obj_from_str, Mixup, preprocess_config, match_partition_rules, \
    get_partition_rules_vit, get_partition_rules_caformer
import os
import jax.numpy as jnp
from convert_model_pytorch import convert_torch_to_flax_conv_next,convert_torch_to_flax_meta_former
import orbax.checkpoint as ocp
from timm.models import MetaFormer,ConvNeXt

# def load_pretrained_params(pretrained_ckpt,abstract_state):
#
#
#
#
#
#
#     checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
#     state = checkpointer.restore(pretrained_ckpt )['model']
#     # params = state['ema_params']
#     params = state['params']
#     return params
#     # jax.tree_util.tree_map(jnp.asarray, params)
#     # print(params.keys())
#     # return {'model': params}

def load_pretrained_params(pretrained_ckpt, abstract_state):
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    ckpt = {'model': abstract_state}
    restore_kwargs = {
        "restore_args": jax.tree_map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray), ckpt
        )
    }
    state = checkpointer.restore(pretrained_ckpt, item=ckpt, **restore_kwargs)['model']
    params = state.ema_params
    return params




def load_pretrain(pretrained_model='convnext_base.fb_in1k',default_params=None):
    model_torch = timm.create_model(pretrained_model, pretrained=True)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")

    if isinstance(model_torch,ConvNeXt):
        model_jax_params = convert_torch_to_flax_conv_next(params, sep='',default_params=default_params)
    elif isinstance(model_torch,MetaFormer):
        model_jax_params = convert_torch_to_flax_meta_former(params, sep='', )
    else:
        raise NotImplemented()

    model_jax_params=jax.tree_util.tree_map(jnp.asarray,model_jax_params)
    return {'model':model_jax_params}



def create_train_state(train_state_config, image_size: int = 224, warmup_steps=1, training_steps=10,
                       grad_accum_steps=1,mesh=None,logical_axis_rules=None
                       ):  # -> TrainState:


    model_config = train_state_config['model']
    optimizer_config = train_state_config['optimizer']
    train_module_config = train_state_config['train_module']
    pretrained_ckpt=train_state_config.pop('pretrained_ckpt',None)

    model = get_obj_from_str(model_config['target'])(**model_config['model_kwargs'])
    print(f'{train_module_config=}')

    train_module = get_obj_from_str(train_module_config.pop('target'))  #(**model_config['model_kwargs'])

    module = train_module(
        model=model,
        mixup=Mixup(train_module_config.pop('mixup',), train_module_config.pop('cutmix')),
        label_smoothing=train_module_config.pop('label_smoothing') if train_module_config['criterion'] != "bce" else 0,
        criterion=CRITERION_COLLECTION[train_module_config.pop('criterion')],**train_module_config
    )
    if jax.process_index() == 0:
        print(module)

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of model and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "images": jnp.zeros((1, 3, image_size, image_size), dtype=jnp.uint8),
        # "labels": jnp.zeros((1,), dtype=jnp.int32),
        "labels": jnp.ones((1,),dtype=jnp.int32)#jnp.array([1,2], dtype=jnp.int32),
    }

    init_rngs = {"params": jax.random.PRNGKey(train_state_config['init_seed'])}

    if jax.process_index()==0:
        print(module.tabulate(init_rngs, **example_inputs,
                              depth=2,compute_flops=True,console_kwargs={'width': 160},
                              compute_vjp_flops=True))

    params = module.init(init_rngs, **example_inputs,det=False)["params"]






    # if args.grad_accum > 1:
    #     grad_accum = jax.tree_map(jnp.zeros_like, params)
    lr = optimizer_config['optimizer_kwargs'].pop('learning_rate')
    end_lr = optimizer_config['optimizer_kwargs'].pop('end_learning_rate',1e-5)
    schedule = optimizer_config['optimizer_kwargs'].pop('schedule','cosine')

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    tx_target = OPTIMIZER_COLLECTION[optimizer_config['target']]
    tx_restore_target = OPTIMIZER_COLLECTION['lamb']

    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32,static_args=('tx_target',))
    def create_optimizer_fn(
            learning_rate: optax.Schedule,tx_target
    ) -> optax.GradientTransformation:
        tx = tx_target(
            learning_rate=learning_rate,
            **optimizer_config['optimizer_kwargs'],
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
        return tx


    if schedule !='cosine':
        learning_rate=lr
    else:
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=end_lr,
        )


    def init_fn(params,tx_target)->TrainState:
        tx = create_optimizer_fn(learning_rate,tx_target)

        if grad_accum_steps > 1:
            grad_accum = jax.tree_map(jnp.zeros_like, params)

        state= TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=tx,
            mixup_rng=jax.random.PRNGKey(train_state_config['mixup_seed']),
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed'] ),
            adv_rng=jax.random.PRNGKey(2036 ),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state

    train_state_shapes = jax.eval_shape(partial(init_fn,tx_target=tx_target), params,)
    train_state_partition = match_partition_rules(get_partition_rules_caformer(), train_state_shapes)
    # jax.sharding.NamedSharding(mesh,train_state_partition)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)


    logical_state_spec = flax.linen.get_partition_spec(train_state_shapes)

    logical_state_sharding = flax.linen.logical_to_mesh_sharding(logical_state_spec, mesh, logical_axis_rules)
    # print(logical_state_sharding)


    # print(train_state_sharding)
    # state=jax.jit(init_fn, #in_shardings=(train_state_partition.params, ),
    #     out_shardings=train_state_sharding,
    #     # donate_argnums=(0, )
    #               )(params)


    init_fn_jited=jax.jit(partial(init_fn,tx_target=tx_target), #in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_sharding,static_argnums=(1,)
        # donate_argnums=(0, )
                  )
    abstract_state = jax.eval_shape(partial(init_fn,tx_target=tx_restore_target), params,)
    # abstract_state=init_fn_jited.eval_shape(params,tx_restore_target)





    if pretrained_ckpt is  None:
        pass
    elif 'gs://' in pretrained_ckpt:
        params = load_pretrained_params(pretrained_ckpt,abstract_state )
    else:
        params = load_pretrain(pretrained_model=pretrained_ckpt,default_params=params)

    params=jax.tree_util.tree_map(jnp.asarray,params)



    state=init_fn_jited(params)



    if jax.process_index()==0:
        print(train_state_config)
        # print(state)

    return state,train_state_partition,train_state_sharding







"""



def create_train_state_restore(train_state_config, image_size: int = 224, warmup_steps=1, training_steps=10,
                       grad_accum_steps=1,mesh=None,logical_axis_rules=None
                       ):  # -> TrainState:


    model_config = train_state_config['model']
    optimizer_config = train_state_config['optimizer']
    train_module_config = train_state_config['train_module']
    pretrained_ckpt=train_state_config.pop('pretrained_ckpt',None)

    model = get_obj_from_str(model_config['target'])(**model_config['model_kwargs'])
    print(f'{train_module_config=}')

    train_module=train_modules.TrainModule


    module = train_module(
        model=model,
        mixup=Mixup(train_module_config.pop('mixup',), train_module_config.pop('cutmix')),
        label_smoothing=train_module_config.pop('label_smoothing') if train_module_config['criterion'] != "bce" else 0,
        criterion=CRITERION_COLLECTION[train_module_config.pop('criterion')],
    )


    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of model and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "images": jnp.zeros((1, 3, image_size, image_size), dtype=jnp.uint8),
        # "labels": jnp.zeros((1,), dtype=jnp.int32),
        "labels": jnp.ones((1,),dtype=jnp.int32)#jnp.array([1,2], dtype=jnp.int32),
    }

    init_rngs = {"params": jax.random.PRNGKey(train_state_config['init_seed'])}

    if jax.process_index()==0:
        print(module.tabulate(init_rngs, **example_inputs,
                              depth=2,compute_flops=True,console_kwargs={'width': 160},
                              compute_vjp_flops=True))

    params = module.init(init_rngs, **example_inputs,det=False)["params"]

    lr = optimizer_config['optimizer_kwargs'].pop('learning_rate')
    end_lr = optimizer_config['optimizer_kwargs'].pop('end_learning_rate',1e-5)
    schedule = optimizer_config['optimizer_kwargs'].pop('schedule','cosine')

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = OPTIMIZER_COLLECTION[optimizer_config['target']](
            learning_rate=learning_rate,
            **optimizer_config['optimizer_kwargs'],
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
        return tx





    if schedule !='cosine':
        learning_rate=lr
    else:
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=end_lr,
        )


    def init_fn(params,tx_target)->TrainState:
        tx = create_optimizer_fn(learning_rate,tx_target)

        if grad_accum_steps > 1:
            grad_accum = jax.tree_map(jnp.zeros_like, params)

        state= TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=tx,
            mixup_rng=jax.random.PRNGKey(train_state_config['mixup_seed']),
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed'] ),
            adv_rng=jax.random.PRNGKey(2036 ),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state

    train_state_shapes = jax.eval_shape(init_fn, params)
    train_state_partition = match_partition_rules(get_partition_rules_caformer(), train_state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)



    state=jax.jit(init_fn, #in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_sharding,
                  )(params)


    if jax.process_index()==0:
        print(train_state_config)
        # print(state)

    return state,train_state_partition,train_state_sharding










"""








if __name__ == "__main__":

    os.environ['GCS_DATASET_DIR']='hello'

    yaml = read_yaml('configs/adv/convnext-b-3step-200ep-ft.yaml')
    yaml = preprocess_config(yaml)

    # print(os.environ.get('GCS_DATASET_DIR'))

    # print(yaml)
    # print(json.dumps(yaml, indent=5))
    #
    # while True:
    #     pass


    state=create_train_state(yaml['train_state'])
    # print(state)
    state=state.replicate()

