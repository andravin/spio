#!/bin/bash

MAX_BATCH_SIZE=${1:-256}


source profile_kernels_with_model.bash

batch_size_convnext_tiny=$((MAX_BATCH_SIZE))
batch_size_convnext_small=$((batch_size_convnext_tiny / 2))
batch_size_convnext_medium=$((batch_size_convnext_small / 2))

extra="--timm-model-kwargs group_size=8 kernel_sizes=5"

profile_kernels_with_model "convnext_tiny"  $batch_size_convnext_tiny $extra
profile_kernels_with_model "convnext_small"  $batch_size_convnext_small $extra
profile_kernels_with_model "convnext_base"  $batch_size_convnext_medium $extra
