python3 src/main.py \
    --output-dir $GCS_MODEL_DIR \
    --train-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar" \
    --valid-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-validation-{00..63}.tar" \
    --train-batch-size 2048 \
    --valid-batch-size 512 \
    --train-loader-workers 40 \
    --valid-loader-workers 10 \
    --random-crop rrc \
    --color-jitter 0.3 \
    --auto-augment "3a" \
    --random-erasing 0.0 \
    --augment-repeats 3 \
    --test-crop-ratio 1.0 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --criterion bce \
    --label-smoothing 0.0 \
    --layers 12 \
    --dim 768 \
    --heads 12 \
    --labels 1000 \
    --layerscale \
    --patch-size 16 \
    --image-size 192 \
    --posemb learnable \
    --pooling cls \
    --dropout 0.0 \
    --droppath 0.2 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --shuffle-seed 0 \
    --optimizer lamb \
    --learning-rate 0.003 \
    --weight-decay 0.05 \
    --adam-b1 0.9 \
    --adam-b2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 1.0 \
    --grad-accum 1 \
    --warmup-steps $((1281167 * 5 / 2048)) \
    --training-steps $((1281167 * 800 / 2048)) \
    --log-interval 100 \
    --eval-interval $((1281167 * 1 / 2048)) \
    --project deit3-jax \
    --name $(basename $0 .sh) \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname)