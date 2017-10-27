#ifndef ALS_ONCUDA
#define ALS_ONCUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "ALS_headerToC.h"

//__device__ void Mt_byM_multiply_k(int i, int j, float**M, float**Result);
__device__ void Mt_byM_multiply_k(int i, int j, float*H, float**Result, const long ptr, const unsigned *idx);
__device__ void inverseMatrix_CholeskyMethod_k(int n, float** A);
__device__ void choldcsl_k(int n, float** A);
__device__ void choldc1_k(int n, float** a, float* p);
__global__ void updateW_overH_kernel(const long rows, const long *row_ptr, const unsigned *col_idx, const unsigned *colMajored_sparse_idx, const float *val, const float lambda, const int k, float *W, float *H);
__global__ void updateH_overW_kernel(const long cols, const long *col_ptr, const unsigned *row_idx, const float *val, const float lambda, const int k, float *W, float *H);



cudaError_t als_NV(smat_t_C_als &R_C, float ** &W, float ** &H, params_als &parameters);

#endif