#!/bin/bash
set -x

make clean -C ./build-release/
make -j8 -C ./build-release/

./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 ../DATASETS/ml10M/
