#!/bin/bash

MAX_BATCH_SIZE=${1:-256}

./efficientnet.bash ${MAX_BATCH_SIZE}

./convnext.bash ${MAX_BATCH_SIZE}
