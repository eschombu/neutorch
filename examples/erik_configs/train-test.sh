#!/bin/bash
#--time 7-12 
module load gcc/11.4.0
#module load cmake

OUT_PATTERN="neutrain-affs-vol_out_$(date +%Y%m%d_%H%M%S)"

srun -C a100 --gpus 2 -p gpu --mem-per-gpu=100g --cpus-per-gpu=8 \
  neutrain-affs-vol -c ./whole_brain_affs_sample2-test.yaml \
  2>&1 | tee $OUT_PATTERN.log

#  -o $OUT_PATTERN

#neutrain-affs-vol -c ./whole_brain_affs_sample2.yaml --debug
