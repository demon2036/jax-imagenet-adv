import copy
import json
from functools import partial

import flax
import jax
import optax
import timm

from pre_define import CRITERION_COLLECTION, OPTIMIZER_COLLECTION
from training import TrainState
from utils import read_yaml, get_obj_from_str, Mixup, preprocess_config
import os
import jax.numpy as jnp
from convert_model_pytorch import convert_torch_to_flax_conv_next
import orbax.checkpoint as ocp


def load_pretrained_params(pretrained_ckpt):
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    state = checkpointer.restore(pretrained_ckpt )['model']
    params = state['ema_params']
    return params
    # jax.tree_util.tree_map(jnp.asarray, params)
    # print(params.keys())
    # return {'model': params}



def load_pretrain(pretrained_model='convnext_base.fb_in1k',default_params=None):
    model_torch = timm.create_model(pretrained_model, pretrained=True)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    model_jax_params = convert_torch_to_flax_conv_next(params, sep='',default_params=default_params)
    model_jax_params=jax.tree_util.tree_map(jnp.asarray,model_jax_params)
    return {'model':model_jax_params}



def create_train_state(train_state_config, image_size: int = 224, warmup_steps=1, training_steps=10,
                       # pretrained_ckpt='gs://brid-center-2b/conv-next-b-128-3step-2000ep-ema',
                        pretrained_ckpt=None
                       ):  # -> TrainState:
    model_config = train_state_config['model']
    optimizer_config = train_state_config['optimizer']
    train_module_config = train_state_config['train_module']
    pretrained_ckpt=train_state_config.pop('pretrained_ckpt',None)

    model = get_obj_from_str(model_config['target'])(**model_config['model_kwargs'])

    train_module = get_obj_from_str(train_module_config.pop('target'))  #(**model_config['model_kwargs'])

    module = train_module(
        model=model,
        mixup=Mixup(train_module_config.pop('mixup',), train_module_config.pop('cutmix')),
        label_smoothing=train_module_config.pop('label_smoothing') if train_module_config['criterion'] == "ce" else 0,
        criterion=CRITERION_COLLECTION[train_module_config.pop('criterion')],**train_module_config
    )
    if jax.process_index() == 0:
        print(module)

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of model and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "images": jnp.zeros((2, 3, image_size, image_size), dtype=jnp.uint8),
        # "labels": jnp.zeros((1,), dtype=jnp.int32),
        "labels": jnp.array([1,2], dtype=jnp.int32),
    }

    init_rngs = {"params": jax.random.PRNGKey(train_state_config['init_seed'])}
    # print(module.tabulate(init_rngs, **example_inputs))

    params = module.init(init_rngs, **example_inputs,det=False)["params"]

    print(f'{pretrained_ckpt=}')
    if pretrained_ckpt is  None:
        pass
    elif 'gs://' in pretrained_ckpt:
        print(1)
        params = load_pretrained_params(pretrained_ckpt )
    else:
        params = load_pretrain(pretrained_model=pretrained_ckpt,default_params=params)
    # if args.grad_accum > 1:
    #     grad_accum = jax.tree_map(jnp.zeros_like, params)
    lr = optimizer_config['optimizer_kwargs'].pop('learning_rate')
    end_lr = optimizer_config['optimizer_kwargs'].pop('end_learning_rate',1e-5)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = OPTIMIZER_COLLECTION[optimizer_config['target']](
            learning_rate=learning_rate,
            **optimizer_config['optimizer_kwargs'],
            # b1=args.adam_b1,
            # b2=args.adam_b2,
            # eps=args.adam_eps,
            # weight_decay=args.weight_decay,
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        # if args.lr_decay < 1.0:
        #     layerwise_scales = {
        #         i: optax.scale(args.lr_decay ** (args.layers - i))
        #         for i in range(args.layers + 1)
        #     }
        #     label_fn = partial(get_layer_index_fn, num_layers=args.layers)
        #     label_fn = partial(tree_map_with_path, label_fn)
        #     tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        # if args.clip_grad > 0:
        #     tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=training_steps,
        end_value=end_lr,
    )

    state= TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer_fn(learning_rate),
        mixup_rng=jax.random.PRNGKey(train_state_config['mixup_seed'] + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed'] + jax.process_index()),
        adv_rng=jax.random.PRNGKey(2036 + jax.process_index()),
        micro_step=0,
        ema_decay=train_state_config['ema_decay'],
        # micro_in_mini=args.grad_accum,
        # grad_accum=grad_accum if args.grad_accum > 1 else None,
        ema_params=copy.deepcopy(params)
    )



    return state


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

