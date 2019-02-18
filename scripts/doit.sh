#!/bin/bash
set -x

make -C ./build-release/

./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 ../DATASETS/toy-example
