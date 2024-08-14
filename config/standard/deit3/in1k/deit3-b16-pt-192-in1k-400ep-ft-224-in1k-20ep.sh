python3 src/main.py \
    --output-dir $GCS_MODEL_DIR \
    --pretrained-ckpt $GCS_MODEL_DIR/deit3-b16-pt-192-in1k-400ep-last.msgpack \
    --train-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar" \
    --valid-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-validation-{00..63}.tar" \
    --train-batch-size 512 \
    --valid-batch-size 512 \
    --train-loader-workers 40 \
    --valid-loader-workers 10 \
    --random-crop rrc \
    --color-jitter 0.0 \
    --auto-augment rand-m9-mstd0.5-inc1 \
    --random-erasing 0.0 \
    --augment-repeats 1 \
    --test-crop-ratio 1.0 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --criterion ce \
    --label-smoothing 0.1 \
    --layers 12 \
    --dim 768 \
    --heads 12 \
    --labels 1000 \
    --layerscale \
    --patch-size 16 \
    --image-size 224 \
    --posemb learnable \
    --pooling cls \
    --dropout 0.0 \
    --droppath 0.1 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --shuffle-seed 0 \
    --optimizer adamw \
    --learning-rate 1e-5 \
    --weight-decay 0.1 \
    --adam-b1 0.9 \
    --adam-b2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 0.0 \
    --grad-accum 1 \
    --warmup-steps $((1281167 * 5 / 512)) \
    --training-steps $((1281167 * 20 / 512)) \
    --log-interval 100 \
    --eval-interval $((1281167 * 1 / 512)) \
    --project deit3-jax \
    --name $(basename $0 .sh) \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname)