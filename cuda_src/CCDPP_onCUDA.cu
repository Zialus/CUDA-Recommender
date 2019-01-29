#include "CCDPP_onCUDA.h"

__global__ void RankOneUpdate_DUAL_kernel(const long Rcols,
                                          const long* Rcol_ptr,
                                          const long* Rrow_idx,
                                          const float* Rval,
                                          float* u,
                                          float* v,
                                          const float lambda,
                                          const int do_nmf,

                                          const long Rcols_t,
                                          const long* Rcol_ptr_t,
                                          const long* Rrow_idx_t,
                                          const float* Rval_t
) {

    long ii = threadIdx.x + blockIdx.x * blockDim.x;


    for (long c = ii; c < Rcols; c += blockDim.x * gridDim.x) {
        v[c] = RankOneUpdate_dev(Rcol_ptr, Rrow_idx, Rval, c, u, lambda * (Rcol_ptr[c + 1] - Rcol_ptr[c]), do_nmf);

    }

    for (long c = ii; c < Rcols_t; c += blockDim.x * gridDim.x) {
        u[c] = RankOneUpdate_dev(Rcol_ptr_t, Rrow_idx_t, Rval_t, c, v, lambda * (Rcol_ptr_t[c + 1] - Rcol_ptr_t[c]), do_nmf);

    }

}

__device__ float RankOneUpdate_dev(const long* Rcol_ptr,
                                   const long* Rrow_idx,
                                   const float* Rval,
                                   const long j,
                                   const float* u_vec_t,

                                   const float lambda,
                                   const int do_nmf
) {

    float g = 0, h = lambda;
    if (Rcol_ptr[j + 1] == Rcol_ptr[j]) { return 0; }
    for (long idx = Rcol_ptr[j]; idx < Rcol_ptr[j + 1]; ++idx) {
        long i = Rrow_idx[idx];
        g += u_vec_t[i] * Rval[idx];
        h += u_vec_t[i] * u_vec_t[i];
    }
    float newvj = g / h;
    if (do_nmf > 0 & newvj < 0) {
        newvj = 0;
    }
    return newvj;
}

__global__ void UpdateRating_DUAL_kernel_NoLoss(const long Rcols,
                                                const long* Rcol_ptr,
                                                const long* Rrow_idx,
                                                float* Rval,
                                                const float* Wt_vec_t,
                                                const float* Ht_vec_t,
                                                const bool add,

                                                const long Rcols_t,
                                                const long* Rcol_ptr_t,
                                                const long* Rrow_idx_t,
                                                float* Rval_t,
                                                const bool add_t
) {
    int ii = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = ii; i < Rcols; i += blockDim.x * gridDim.x) {
        if (add) {
            float Htc = Ht_vec_t[i];
            for (long idx = Rcol_ptr[i]; idx < Rcol_ptr[i + 1]; ++idx) {
                Rval[idx] += Wt_vec_t[Rrow_idx[idx]] * Htc; //change R.val
            }
        } else {
            float Htc = Ht_vec_t[i];
            for (long idx = Rcol_ptr[i]; idx < Rcol_ptr[i + 1]; ++idx) {
                Rval[idx] -= Wt_vec_t[Rrow_idx[idx]] * Htc; //change R.val
            }
        }
    }

    for (int i = ii; i < Rcols_t; i += blockDim.x * gridDim.x) {
        if (add_t) {
            float Htc = Wt_vec_t[i];
            for (long idx = Rcol_ptr_t[i]; idx < Rcol_ptr_t[i + 1]; ++idx) {
                Rval_t[idx] += Ht_vec_t[Rrow_idx_t[idx]] * Htc; //change R.val
            }
        } else {
            float Htc = Wt_vec_t[i];
            for (long idx = Rcol_ptr_t[i]; idx < Rcol_ptr_t[i + 1]; ++idx) {
                Rval_t[idx] -= Ht_vec_t[Rrow_idx_t[idx]] * Htc; //change R.val
            }
        }
    }
}

void kernel_wrapper_ccdpp_NV(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters) {
    cudaError_t cudaStatus;
    cudaStatus = ccdpp_NV(R, T, W, H, parameters);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ALS FAILED: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceReset();
    gpuErrchk(cudaStatus);
}

// Helper function for using CUDA.
cudaError_t ccdpp_NV(smat_t& R_C, testset_t& T, mat_t& W, mat_t& H, parameter& parameters) {
    long* dev_Rcol_ptr = nullptr;
    long* dev_Rrow_idx = nullptr;
    long* dev_Rcol_ptr_T = nullptr;
    long* dev_Rrow_idx_T = nullptr;
    float* dev_Rval = nullptr;
    float* dev_Rval_t = nullptr;
    float* dev_Wt_vec_t = nullptr; //u
    float* dev_Ht_vec_t = nullptr; //v


    unsigned nThreadsPerBlock = parameters.nThreadsPerBlock;
    unsigned nBlocks = parameters.nBlocks;
    cudaError_t cudaStatus;

    int k = parameters.k;
    float lambda = parameters.lambda;

    // Create transpose view of R
    smat_t Rt;
    Rt = R_C.transpose();

    // H is a zero matrix now.
    for (int t = 0; t < k; ++t) { for (long c = 0; c < R_C.cols; ++c) { H[t][c] = 0; }}

    size_t nbits_u = R_C.rows * sizeof(float);
    size_t nbits_v = R_C.cols * sizeof(float);

    // Reset GPU.
    cudaStatus = cudaDeviceReset();
    gpuErrchk(cudaStatus);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    gpuErrchk(cudaStatus);

    // Allocate GPU buffers for all vectors.
    cudaStatus = cudaMalloc((void**) &dev_Rcol_ptr, R_C.nbits_col_ptr);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Rrow_idx, R_C.nbits_row_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Rcol_ptr_T, Rt.nbits_col_ptr);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Rrow_idx_T, Rt.nbits_row_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Rval, R_C.nbits_val);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Rval_t, Rt.nbits_val);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Wt_vec_t, nbits_u);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_Ht_vec_t, nbits_v);
    gpuErrchk(cudaStatus);

    // Copy all vectors to GPU buffers.
    cudaStatus = cudaMemcpy(dev_Rval, R_C.val, R_C.nbits_val, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rval_t, Rt.val, Rt.nbits_val, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rcol_ptr, R_C.col_ptr, R_C.nbits_col_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rrow_idx, R_C.row_idx, R_C.nbits_row_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rcol_ptr_T, Rt.col_ptr, Rt.nbits_col_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rrow_idx_T, Rt.row_idx, Rt.nbits_row_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);


    for (int oiter = 1; oiter <= parameters.maxiter; ++oiter) {

        for (int t = 0; t < k; ++t) {

            float* Wt = &W[t][0];
            float* Ht = &H[t][0];

            cudaStatus = cudaMemcpy(dev_Wt_vec_t, Wt, nbits_u, cudaMemcpyHostToDevice);
            gpuErrchk(cudaStatus);
            cudaStatus = cudaMemcpy(dev_Ht_vec_t, Ht, nbits_v, cudaMemcpyHostToDevice);
            gpuErrchk(cudaStatus);

            if (oiter > 1) {
                UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx,
                        dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, true, Rt.cols,
                        dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, true);

                cudaStatus = cudaDeviceSynchronize();
                gpuErrchk(cudaStatus);
            }


            for (int iter = 1; iter <= parameters.maxinneriter; ++iter) {
                RankOneUpdate_DUAL_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx,
                        dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, lambda, parameters.do_nmf,
                        Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t);

                cudaStatus = cudaDeviceSynchronize();
                gpuErrchk(cudaStatus);
            }

            cudaStatus = cudaMemcpy(Wt, dev_Wt_vec_t, nbits_u, cudaMemcpyDeviceToHost);
            gpuErrchk(cudaStatus);
            cudaStatus = cudaMemcpy(Ht, dev_Ht_vec_t, nbits_v, cudaMemcpyDeviceToHost);
            gpuErrchk(cudaStatus);

            UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx,
                    dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, false, Rt.cols,
                    dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, false);

            cudaStatus = cudaDeviceSynchronize();
            gpuErrchk(cudaStatus);
        }
    }


    cudaFree(dev_Rcol_ptr);
    cudaFree(dev_Rrow_idx);
    cudaFree(dev_Rcol_ptr_T);
    cudaFree(dev_Rrow_idx_T);
    cudaFree(dev_Rval);
    cudaFree(dev_Rval_t);
    cudaFree(dev_Wt_vec_t);
    cudaFree(dev_Ht_vec_t);

    return cudaStatus;
}

