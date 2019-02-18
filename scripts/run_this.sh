#!/bin/bash
set -x

./exec/cuda_andre -CUDA -OMP -ALS -nBlocks 16 -q 1 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/toy-example/
