#!/bin/bash

# ConvFirst
# python profile_block.py  --group-width 8 --channels 64 --kernel-size 3 --resolution 64 --depth 4 --extra 2 --block ConvFirst --batch-size 64 --expansion-ratio 6 --spio --benchmark-configs --max-random-samples 0
python profile_block.py  --group-width 8 --channels 64 --kernel-size 3 --resolution 64 --depth 4 --extra 2 --block ConvFirst --batch-size 256 --expansion-ratio 6 --spio --benchmark-configs --max-random-samples 0


# MBConv
# python profile_block.py  --group-width 8 --channels 256 --kernel-size 3 --resolution 16 --depth 4 --extra 2 --block MBConv --batch-size 64 --expansion-ratio 4 --spio --benchmark-configs --max-random-samples 0
python profile_block.py  --group-width 8 --channels 256 --kernel-size 3 --resolution 16 --depth 4 --extra 2 --block MBConv --batch-size 256 --expansion-ratio 4 --spio --benchmark-configs --max-random-samples 0
