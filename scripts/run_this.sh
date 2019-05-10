#!/bin/bash
set -x

./exec/cuda_andre -CUDA -OMP -ALS -nBlocks 32 -q 1 -l 0.05 -n 8 -k 5 -t 15 ../DATASETS/netflix/
