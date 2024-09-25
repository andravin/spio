# TODO: Is one epoch accurate enough for the perf model?
EPOCHS=${1:-"1"}

export SPIO_WORKERS=`nproc`

cd ~/spio/scripts

time python benchmark.py --kernel Conv2dGw8Kernel --params-set --full-format --ssv --write-to-file --randomize-batch-size --epochs=$EPOCHS
time python benchmark.py --kernel Conv2dGw8Kernel --params-set --full-format --ssv --write-to-file --randomize-batch-size --epochs=$EPOCHS --kernel-kwargs igrad=True
time python benchmark.py --kernel Conv2dGw8WgradKernel --params-set --full-format --ssv --write-to-file --randomize-batch-size --epochs=$EPOCHS
