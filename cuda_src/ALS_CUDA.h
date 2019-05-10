#ifndef ALS_CUDA_H
#define ALS_CUDA_H

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "CUDA_AUX.h"
#include "pmf.h"

__device__ void choldc1_k(int n, float* a, float* p);

__device__ void choldcsl_k(int n, float* A);

__device__ void inverseMatrix_CholeskyMethod_k(int n, float* A);

__device__ void Mt_byM_multiply_k(long i, long j, float* H, float* Result, long ptr, const unsigned * idx);

__global__ void updateW_overH_kernel(long rows,
                                     const unsigned* row_ptr,
                                     const unsigned* col_idx,
                                     const float* val_t,
                                     const float* val,
                                     float lambda,
                                     unsigned k,
                                     float* W,
                                     float* H);

__global__ void updateH_overW_kernel(long cols,
                                     const unsigned* col_ptr,
                                     const unsigned* row_idx,
                                     const float* val,
                                     float lambda,
                                     unsigned k,
                                     float* W,
                                     float* H);

void kernel_wrapper_als_NV(SparseMatrix& R, TestData& T, MatData& W, MatData& H, parameter& parameters);

cudaError_t als_NV(SparseMatrix& R, TestData& T, MatData& W, MatData& H, parameter& parameters);

#endif //ALS_CUDA_H
