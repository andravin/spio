#!/bin/bash

MODEL=$1
BATCH_SIZE=$2
BENCH=${3:-train}
EXTRA_MODEL_KWARGS=${4:-""}
SKIP_COMPILE=${5:-"false"}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> <batch_size> [train|inference]"
    exit 1
fi

if [[ "$SKIP_COMPILE" == "false" ]]
then
    SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --spio --torchcompile --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
    SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile
    SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
fi

SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --spio --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 $EXTRA_MODEL_KWARGS
