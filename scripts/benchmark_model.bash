#!/bin/bash

# Run this script from pytorch-image-models root directory.
# It uses use the timm benchmark.py script to benchmark a model with and without spio.

MODEL=$1
BATCH_SIZE=${2:-"256"}
BENCH=${3:-train}
EXTRA_ARGS=${4:-""}
SKIP_COMPILE=${5:-"false"}
LOG_LEVEL=${6:-"1"}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> <batch_size> [train|inference] [EXTRA_ARGS] [SKIP_COMPILE] [LOG_LEVEL]"
    exit 1
fi

if [[ "$SKIP_COMPILE" == "false" ]]
then
    SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --spio --torchcompile $EXTRA_ARGS
    SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile $EXTRA_ARGS
fi

SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --spio $EXTRA_ARGS
SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE $EXTRA_ARGS
