#ifndef CCD_CUDA_H
#define CCD_CUDA_H

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "CUDA_AUX.h"
#include "pmf.h"

__global__ void RankOneUpdate_v_kernel(unsigned Rcols,
                                       const unsigned* Rcol_ptr,
                                       const unsigned* Rrow_idx,
                                       const float* Rval,

                                       float* u,
                                       float* v,
                                       float lambda
);

__global__ void RankOneUpdate_u_kernel(unsigned Rcols_t,
                                       const unsigned* Rcol_ptr_t,
                                       const unsigned* Rrow_idx_t,
                                       const float* Rval_t,

                                       float* u,
                                       float* v,
                                       float lambda
);

__global__ void UpdateRating_DUAL_kernel_NoLoss(unsigned Rcols,
                                                const unsigned* Rcol_ptr,
                                                const unsigned* Rrow_idx,
                                                float* Rval,
                                                const float* Wt_vec_t,
                                                const float* Ht_vec_t,
                                                bool add,

                                                unsigned Rcols_t,
                                                const unsigned* Rcol_ptr_t,
                                                const unsigned* Rrow_idx_t,
                                                float* Rval_t,
                                                bool add_t
);

void kernel_wrapper_ccdpp_NV(SparseMatrix& R, TestData& T, MatData& W, MatData& H, parameter& parameters);

cudaError_t ccdpp_NV(SparseMatrix& R, TestData& T, MatData& W, MatData& H, parameter& parameters);

#endif //CCD_CUDA_H
