python3 src/main.py \
    --output-dir $GCS_MODEL_DIR \
    --train-dataset-shards "$GCS_DATASET_DIR/imagenet-w21-wds/imagenet_w21-train-{0000..2047}.tar" \
    --train-batch-size 2048 \
    --train-loader-workers 40 \
    --random-crop src \
    --color-jitter 0.3 \
    --auto-augment "3a" \
    --random-erasing 0.0 \
    --augment-repeats 1 \
    --test-crop-ratio 1.0 \
    --mixup 0.0 \
    --cutmix 1.0 \
    --criterion ce \
    --label-smoothing 0.1 \
    --layers 32 \
    --dim 1280 \
    --heads 16 \
    --labels 19167 \
    --layerscale \
    --patch-size 14 \
    --image-size 126 \
    --posemb learnable \
    --pooling cls \
    --dropout 0.0 \
    --droppath 0.5 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --shuffle-seed 0 \
    --optimizer lamb \
    --learning-rate 0.003 \
    --weight-decay 0.02 \
    --adam-b1 0.9 \
    --adam-b2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 1.0 \
    --grad-accum 1 \
    --warmup-steps $((13151276 * 5 / 2048)) \
    --training-steps $((13151276 * 90 / 2048)) \
    --log-interval 100 \
    --eval-interval $((13151276 * 10 / 2048)) \
    --project deit3-jax \
    --name $(basename $0 .sh) \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname)
