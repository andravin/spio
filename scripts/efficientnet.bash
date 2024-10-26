#!/bin/bash

MAX_BATCH_SIZE=${1:-256}


source profile_kernels_with_model.bash


batch_size_b0=$((MAX_BATCH_SIZE))
batch_size_b1=$((batch_size_b0))
batch_size_b2=$((batch_size_b0 / 2))
batch_size_b3=$((batch_size_b2))
batch_size_b4=$((batch_size_b3 / 2))


extra="--timm-model-kwargs group_size=8"

profile_kernels_with_model "efficientnet_b0"  $batch_size_b0 $extra
profile_kernels_with_model "efficientnet_b1"  $batch_size_b1 $extra
profile_kernels_with_model "efficientnet_b2"  $batch_size_b2 $extra
profile_kernels_with_model "efficientnet_b3"  $batch_size_b3 $extra
profile_kernels_with_model "efficientnet_b4"  $batch_size_b4 $extra

