#!/bin/bash

python profile_block.py --channels 128 --kernel-size 7 --group-width 1 --resolution 64 --depth 4 --extra 2 --block ConvNeXt --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --spio --inference
python profile_block.py --channels 128 --kernel-size 7 --group-width 1 --resolution 64 --depth 4 --extra 2 --block ConvNeXt --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --inference


python profile_block.py --channels 256 --kernel-size 7 --group-width 1 --resolution 16 --depth 4 --extra 2 --block ConvNeXt --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --spio --inference
python profile_block.py --channels 256 --kernel-size 7 --group-width 1 --resolution 16 --depth 4 --extra 2 --block ConvNeXt --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --inference

python profile_block.py --channels 512 --kernel-size 7 --group-width 1 --resolution 16 --depth 4 --extra 2 --block ConvNeXt --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --spio --inference
python profile_block.py --channels 512 --kernel-size 7 --group-width 1 --resolution 16 --depth 4 --extra 2 --block ConvNeXt --batch-start 32 --batch-end 288 --batch-step 32 --expansion-ratio 4 --inference