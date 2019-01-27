#ifndef CCDPP_ONCUDA
#define CCDPP_ONCUDA

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "ERROR_CHECKING.h"
#include "pmf.h"

__global__ void RankOneUpdate_DUAL_kernel(long Rcols,
                                          const long* Rcol_ptr,
                                          const unsigned int* Rrow_idx,
                                          const float* Rval,
                                          float* u,
                                          float* v,
                                          float lambda,
                                          float* innerfundec_cur,
                                          int do_nmf,

                                          long Rcols_t,
                                          const long* Rcol_ptr_t,
                                          const unsigned int* Rrow_idx_t,
                                          const float* Rval_t,
                                          float* innerfundec_cur2
);

__device__ float RankOneUpdate_dev(const long* Rcol_ptr,
                                   const unsigned* Rrow_idx,
                                   const float* Rval,
                                   int j,
                                   const float* u_vec_t,

                                   float lambda,
                                   float vj,
                                   float* redvar,
                                   int do_nmf);

__global__ void UpdateRating_DUAL_kernel_NoLoss(long Rcols,
                                                const long* Rcol_ptr,
                                                const unsigned int* Rrow_idx,
                                                float* Rval,
                                                const float* Wt_vec_t,
                                                const float* Ht_vec_t,
                                                bool add,

                                                long Rcols_t,
                                                const long* Rcol_ptr_t,
                                                const unsigned int* Rrow_idx_t,
                                                float* Rval_t,
                                                bool add_t
);


cudaError_t ccdpp_NV(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters);

void kernel_wrapper_ccdpp_NV(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters);

#endif
