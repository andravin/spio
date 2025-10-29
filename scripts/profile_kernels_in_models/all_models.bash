#!/bin/bash

MAX_BATCH_SIZE=${1:-256}

./efficientnet.bash ${MAX_BATCH_SIZE}

./convnext_gw8_ks5.bash ${MAX_BATCH_SIZE}
