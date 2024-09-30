#!/bin/bash
MODEL=$1
BATCH_SIZE=$2
BENCH=${3:-train}

SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 --spio --torchcompile --torchcompile-mode=reduce-overhead
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile  --torchcompile-mode=reduce-overhead
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 --torchcompile --torchcompile-mode=reduce-overhead

SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 --spio --torchcompile --torchcompile-mode=default
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE --torchcompile  --torchcompile-mode=default
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 --torchcompile --torchcompile-mode=default

SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8 --spio
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE
SPIO_LOGGER=1 python3 benchmark.py --model $MODEL --amp --channels-last --bench=$BENCH   --batch-size=$BATCH_SIZE  --model-kwargs group_size=8
