#ifndef CCDPP_ONCUDA
#define CCDPP_ONCUDA

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>

struct params {
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

struct smat_t_C {
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
};

__global__ void RankOneUpdate_DUAL_kernel(const long Rcols, //are the iterations on for
	const long *Rcol_ptr,
	const unsigned int *Rrow_idx,
	const float *Rval,
	float * u,
	float *v,
	const float lambda,
	float *innerfundec_cur,
	const int do_nmf,

	const long Rcols_t,
	const long *Rcol_ptr_t,
	const unsigned int *Rrow_idx_t,
	const float *Rval_t,
	float *innerfundec_cur2
	);

__device__ float RankOneUpdate_dev(const long *Rcol_ptr,
	const unsigned *Rrow_idx,
	const float *Rval,
	const int j,
	const float * u_vec_t,

	const float lambda,
	const float vj,
	float *redvar,
	const int do_nmf);

__global__ void UpdateRating_DUAL_kernel_NoLoss(const long Rcols, //are the iterations on for
	const long *Rcol_ptr,
	const unsigned int *Rrow_idx,
	float *Rval,
	const float * Wt_vec_t,
	const float * Ht_vec_t,
	const bool add,

	const long Rcols_t, //are the iterations on for
	const long *Rcol_ptr_t,
	const unsigned int *Rrow_idx_t,
	float *Rval_t,
	const bool add_t
	);


cudaError_t ccdpp_NV(smat_t_C& R_C, float**& W, float**& H, params& parameters);

void kernel_wrapper_ccdpp_NV(smat_t_C &R_C, float ** &W, float ** &H, params &parameters);

smat_t_C transpose(smat_t_C m);

float maxC(float a, float b);

#endif
