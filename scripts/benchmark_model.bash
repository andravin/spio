#!/bin/bash

# Run this script from pytorch-image-models root directory.
# It uses use the timm benchmark.py script to benchmark a model with and without spio.

MODEL=$1
BATCH_SIZE=$2
BENCH=${3:-train}
EXTRA_MODEL_KWARGS=${4:-""}
SKIP_COMPILE=${5:-"false"}
LOG_LEVEL=${6:-"1"}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> <batch_size> [train|inference] [EXTRA_MODEL_KWARGS] [SKIP_COMPILE] [LOG_LEVEL]"
    exit 1
fi

if [[ "$SKIP_COMPILE" == "false" ]]
then
    SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --spio --torchcompile --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
    SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile
    SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
fi

SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --spio --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE
SPIO_LOGGER=$LOG_LEVEL python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
