#!/bin/bash
set -x

for n in {1..10}; do
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/ml10M/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 -ALS ../DATASETS/ml10M/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/ml20M/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 -ALS ../DATASETS/ml20M/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/netflix
    ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 -ALS ../DATASETS/netflix/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/jester
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 -ALS ../DATASETS/jester/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/yahoor1/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 -ALS ../DATASETS/yahoor1/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 ../DATASETS/yahooc15/
    # ./exec/cuda_andre -CUDA -OMP -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -n 8 -k 5 -t 15 -ALS ../DATASETS/yahooc15/
done
