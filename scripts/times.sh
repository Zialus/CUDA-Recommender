#!/bin/bash
set -x


T=(1 3 5 7)
K=(1 5 10 15 20 25 30 40 50)

output_dir="results"


for ks in "${K[@]}"; do
    for ts in "${T[@]}"; do
        for n in {1..3}; do
            out_file=${output_dir}/CCD/jester/k-${ks}_T-${ts}_out${n}.txt
            ./exec/cuda_andre -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -T ${ts} -k ${ks} -t 20 ../DATASETS/jester/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/CCD/ml10M/k-${ks}_T-${ts}_out${n}.txt
            ./exec/cuda_andre -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -T ${ts} -k ${ks} -t 20 ../DATASETS/ml10M/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/CCD/ml20M/k-${ks}_T-${ts}_out${n}.txt
            ./exec/cuda_andre -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -T ${ts} -k ${ks} -t 20 ../DATASETS/ml20M/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/CCD/netflix/k-${ks}_T-${ts}_out${n}.txt
            ./exec/cuda_andre -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -T ${ts} -k ${ks} -t 15 ../DATASETS/netflix/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/CCD/yahoor1/k-${ks}_T-${ts}_out${n}.txt
            ./exec/cuda_andre -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -T ${ts} -k ${ks} -t 10 ../DATASETS/yahoor1/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/CCD/yahooc15/k-${ks}_T-${ts}_out${n}.txt
            ./exec/cuda_andre -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -T ${ts} -k ${ks} -t 10 ../DATASETS/yahooc15/ >> ${out_file}
        done
    done
done


for ks in "${K[@]}"; do
        for n in {1..3}; do
            out_file=${output_dir}/ALS/jester/k-${ks}_out${n}.txt
            ./exec/cuda_andre -ALS -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -k ${ks} -t 20 ../DATASETS/jester/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/ALS/ml10M/k-${ks}_out${n}.txt
            ./exec/cuda_andre -ALS -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -k ${ks} -t 20 ../DATASETS/ml10M/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/ALS/ml20M/k-${ks}_out${n}.txt
            ./exec/cuda_andre -ALS -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -k ${ks} -t 20 ../DATASETS/ml20M/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/ALS/netflix/k-${ks}_out${n}.txt
            ./exec/cuda_andre -ALS -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -k ${ks} -t 15 ../DATASETS/netflix/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/ALS/yahoor1/k-${ks}_out${n}.txt
            ./exec/cuda_andre -ALS -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -k ${ks} -t 10 ../DATASETS/yahoor1/ >> ${out_file}
        done
        for n in {1..3}; do
            out_file=${output_dir}/ALS/yahooc15/k-${ks}_out${n}.txt
            ./exec/cuda_andre -ALS -CUDA -nBlocks 32 -nThreadsPerBlock 512 -l 0.1 -k ${ks} -t 10 ../DATASETS/yahooc15/ >> ${out_file}
        done
done
