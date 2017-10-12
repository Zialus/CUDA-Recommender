#ifndef CCDPP_ONCUDA
#define CCDPP_ONCUDA

//#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <stdio.h>
#include <cstdlib>
#include "ccd_headerToC.h"

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


cudaError_t ccdpp_NV(smat_t_C &R_C, float ** &W, float ** &H, params &parameters); 

#endif