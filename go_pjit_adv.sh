#echo "${SCRIPT_PATHS[@]}"
SCRIPT_PATHS=("$@")

#echo "Array content: ${SCRIPT_PATHS[@]}"
for script in "${SCRIPT_PATHS[@]}"; do
    echo " $script"
    pkill -9 -f python
    sudo rm /tmp/libtpu_lockfile
    source ~/miniconda3/bin/activate base;
#    python -u test.py
#    export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=false --xla_tpu_data_parallel_opt_different_sized_ops=false --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
#    export LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=true --xla_enable_async_all_gather=true"
    export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


    python -u train_adv_pjit.py --yaml-path $script
#    python -u main.py --yaml-path $script
#    python -u main_test.py --yaml-path $script

done
