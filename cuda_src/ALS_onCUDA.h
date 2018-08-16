#ifndef ALS_ONCUDA
#define ALS_ONCUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

struct params_als {
	int nBlocks;
	int nThreadsPerBlock;
	int k;
	int maxiter;
	int inneriter;
	int do_nmf;
	int verbose;
	float lambda;
	float eps;
	bool enable_cuda;
};
struct smat_t_C_als {
	long rows, cols;
	long nnz, max_row_nnz, max_col_nnz;
	float *val, *val_t;
	size_t nbits_val, nbits_val_t;
	float *weight, *weight_t;
	size_t nbits_weight, nbits_weight_t;
	long *col_ptr, *row_ptr;
	size_t nbits_col_ptr, nbits_row_ptr;
	//long *col_nnz, *row_nnz;
	size_t nbits_col_nnz, nbits_row_nnz;
	unsigned *row_idx, *col_idx;
	size_t nbits_row_idx, nbits_col_idx;
	bool with_weights;
	unsigned *colMajored_sparse_idx;
	size_t nbits_colMajored_sparse_idx;
};

void Mt_byM_multiply_(int i, int j, float**M, float**Result);
void inverseMatrix_CholeskyMethod_(int n, float** A);
void choldcsl_(int n, float** A);
void choldc1_(int n, float** a, float* p);

void Mt_byM_multiply_off(int i, int j, float*H, float**Result, const long ptr, const unsigned *idx);
void inverseMatrix_CholeskyMethod_off(int n, float* A);
void choldcsl_off(int n, float* A);
void choldc1_off(int n, float* a, float* p);


__device__ void choldc1_k(int n, float** a, float* p);
__device__ void choldcsl_k(int n, float** A);
__device__ void inverseMatrix_CholeskyMethod_k(int n, float** A);

__device__ void Mt_byM_multiply_k(int i, int j, float*H, float**Result, const long ptr, const unsigned *idx);

__global__ void updateW_overH_kernel(const long rows, const long *row_ptr, const unsigned *col_idx, const unsigned *colMajored_sparse_idx, const float *val, const float lambda, const int k, float *W, float *H);
__global__ void updateH_overW_kernel(const long cols, const long *col_ptr, const unsigned *row_idx, const float *val, const float lambda, const int k, float *W, float *H);

void kernel_wrapper_als_NV(smat_t_C_als &R_C, float ** &W, float ** &H, params_als &parameters);

//__device__ void Mt_byM_multiply_k(int i, int j, float**M, float**Result);


cudaError_t als_NV(smat_t_C_als &R_C, float ** &W, float ** &H, params_als &parameters);

#endif