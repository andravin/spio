#!/bin/bash

# ConvFirst
python profile_block.py  --group-width 8 --channels 64 --kernel-size 3 --resolution 64 --depth 4 --extra 2 --block ConvFirst --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 6 --spio
python profile_block.py  --group-width 8 --channels 64 --kernel-size 3 --resolution 64 --depth 4 --extra 2 --block ConvFirst --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 6


# MBConv
python profile_block.py  --group-width 8 --channels 256 --kernel-size 3 --resolution 16 --depth 4 --extra 2 --block MBConv --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --spio
python profile_block.py  --group-width 8 --channels 256 --kernel-size 3 --resolution 16 --depth 4 --extra 2 --block MBConv --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4
