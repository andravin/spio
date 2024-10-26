#!/bin/bash

SCRIPTS_DIR=..

function profile_kernels_with_model() {
    model=$1
    shift
    max_batch_size=$1
    shift
    extra=$@
    for batch_size in 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512
    do
        if [ $batch_size -gt $max_batch_size ]
        then
            break
        fi
        echo "Benchmarking ${model} with batch size ${batch_size} .."
        time python $SCRIPTS_DIR/profile_block.py --batch-size=${batch_size} --timm-model=${model} --spio --benchmark-configs ${extra}
        echo ".. done."
        echo
    done
}
