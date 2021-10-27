#!/bin/bash
export NCCL_IB_DISABLE=1
PYTHONPATH=./:$PYTHONPATH OMP_NUM_THREADS=1 python3 ../tools/train_net.py --num-gpus 8
