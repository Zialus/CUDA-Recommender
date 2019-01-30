#ifndef ALS_ONCUDA
#define ALS_ONCUDA

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "CUDA_AUX.h"
#include "pmf.h"

__device__ void choldc1_k(int n, float** a, float* p);

__device__ void choldcsl_k(int n, float** A);

__device__ void inverseMatrix_CholeskyMethod_k(int n, float** A);

__device__ void Mt_byM_multiply_k(long i, long j, float* H, float** Result, long ptr, const long* idx);

__global__ void updateW_overH_kernel(long rows,
                                     const long* row_ptr,
                                     const long* col_idx,
                                     const long* colMajored_sparse_idx,
                                     const float* val,
                                     float lambda,
                                     unsigned k,
                                     float* W,
                                     float* H);

__global__ void updateH_overW_kernel(long cols,
                                     const long* col_ptr,
                                     const long* row_idx,
                                     const float* val,
                                     float lambda,
                                     unsigned k,
                                     float* W,
                                     float* H);

void kernel_wrapper_als_NV(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters);

cudaError_t als_NV(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters);

#endif
