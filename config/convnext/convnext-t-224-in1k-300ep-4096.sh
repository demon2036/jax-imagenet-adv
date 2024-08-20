export  TRAIN_BATCH_SIZE=4096

python3  main.py \
    --output-dir $GCS_MODEL_DIR \
    --train-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar" \
    --valid-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-validation-{00..63}.tar" \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --valid-batch-size 512 \
    --train-loader-workers 40 \
    --valid-loader-workers 10 \
    --random-crop rrc \
    --color-jitter 0.0 \
    --auto-augment rand-m9-mstd0.5-inc1 \
    --random-erasing 0.25 \
    --augment-repeats 3 \
    --test-crop-ratio 0.875 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --criterion ce \
    --label-smoothing 0.1 \
    --layers 12 \
    --dim 768 \
    --heads 12 \
    --labels 1000 \
    --patch-size 16 \
    --image-size 224 \
    --posemb learnable \
    --pooling cls \
    --dropout 0.0 \
    --droppath 0.05 \
    --init-seed 1 \
    --mixup-seed 1 \
    --dropout-seed 1 \
    --shuffle-seed 1 \
    --optimizer adamw \
    --learning-rate 0.004 \
    --weight-decay 0.05 \
    --adam-b1 0.9 \
    --adam-b2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 0.0 \
    --grad-accum 1 \
    --warmup-steps $((1281167 * 20 / $TRAIN_BATCH_SIZE)) \
    --training-steps $((1281167 * 300 / $TRAIN_BATCH_SIZE)) \
    --log-interval 100 \
    --eval-interval $((1281167 * 10 / $TRAIN_BATCH_SIZE)) \
    --project deit3-jax \
    --name $(basename $0 .sh) \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname)
