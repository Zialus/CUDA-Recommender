#include "CCDPP_onCUDA.h"

__global__ void RankOneUpdate_DUAL_kernel(const long Rcols,
                                          const long* Rcol_ptr,
                                          const unsigned int* Rrow_idx,
                                          const float* Rval,
                                          float* u,
                                          float* v,
                                          const float lambda,
                                          float* innerfundec_cur,
                                          const int do_nmf,

                                          const long Rcols_t,
                                          const long* Rcol_ptr_t,
                                          const unsigned int* Rrow_idx_t,
                                          const float* Rval_t,
                                          float* innerfundec_cur2
) {

    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    float dev_innerfundec_cur = 0;
    float dev_innerfundec_cur2 = 0;
    innerfundec_cur[ii] = 0;
    innerfundec_cur2[ii] = 0;
    for (int c = ii; c < Rcols; c += blockDim.x * gridDim.x) {
        v[c] = RankOneUpdate_dev(Rcol_ptr, Rrow_idx, Rval,
                                 c, u, lambda * (Rcol_ptr[c + 1] - Rcol_ptr[c]), v[c], &dev_innerfundec_cur, do_nmf);

    }
    innerfundec_cur[ii] = dev_innerfundec_cur;

    for (int c = ii; c < Rcols_t; c += blockDim.x * gridDim.x) {
        u[c] = RankOneUpdate_dev(Rcol_ptr_t, Rrow_idx_t, Rval_t,
                                 c, v, lambda * (Rcol_ptr_t[c + 1] - Rcol_ptr_t[c]), u[c], &dev_innerfundec_cur2, do_nmf);

    }
    innerfundec_cur2[ii] = dev_innerfundec_cur2;

}

__device__ float RankOneUpdate_dev(const long* Rcol_ptr,
                                   const unsigned* Rrow_idx,
                                   const float* Rval,
                                   const int j,
                                   const float* u_vec_t,

                                   const float lambda,
                                   const float vj,
                                   float* redvar,
                                   const int do_nmf) {

    float g = 0, h = lambda;
    if (Rcol_ptr[j + 1] == Rcol_ptr[j]) { return 0; }
    for (long idx = Rcol_ptr[j]; idx < Rcol_ptr[j + 1]; ++idx) {
        int i = Rrow_idx[idx];
        g += u_vec_t[i] * Rval[idx];
        h += u_vec_t[i] * u_vec_t[i];
    }
    float newvj = g / h, delta = 0, fundec = 0;
    if (do_nmf > 0 & newvj < 0) {
        newvj = 0;
        delta = vj; // old - new
        fundec = -2 * g * vj; //+h*vj*vj;
    } else {
        delta = vj - newvj;
        fundec = h * delta * delta;
    }
    *redvar += fundec;
    return newvj;
}

__global__ void UpdateRating_DUAL_kernel_NoLoss(const long Rcols,
                                                const long* Rcol_ptr,
                                                const unsigned int* Rrow_idx,
                                                float* Rval,
                                                const float* Wt_vec_t,
                                                const float* Ht_vec_t,
                                                const bool add,

                                                const long Rcols_t,
                                                const long* Rcol_ptr_t,
                                                const unsigned int* Rrow_idx_t,
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
    unsigned* dev_Rrow_idx = nullptr;
    long* dev_Rcol_ptr_T = nullptr;
    unsigned* dev_Rrow_idx_T = nullptr;
    float* dev_Rval = nullptr;
    float* dev_Rval_t = nullptr;
    float* dev_Wt_vec_t = nullptr;
    float* dev_Ht_vec_t = nullptr;

    float* dev_return = nullptr;
    float* dev_return2 = nullptr;
    float* Hostreduction = nullptr;
    float* Hostreduction2 = nullptr;


    unsigned nThreadsPerBlock = parameters.nThreadsPerBlock;
    unsigned nBlocks = parameters.nBlocks;
    cudaError_t cudaStatus;
    Hostreduction = (float*) malloc(sizeof(float) * (nThreadsPerBlock * nBlocks));
    Hostreduction2 = (float*) malloc(sizeof(float) * (nThreadsPerBlock * nBlocks));


    int k = parameters.k;
    int maxiter = parameters.maxiter;
    int inneriter = parameters.maxinneriter;
    float lambda = parameters.lambda;
    float eps = parameters.eps;
    long num_updates = 0;


    // Create transpose view of R
    smat_t Rt;
    Rt = R_C.transpose();
    // initial value of the regularization term
    // H is a zero matrix now.
    for (int t = 0; t < k; ++t) { for (long c = 0; c < R_C.cols; ++c) { H[t][c] = 0; }}

    float* u, * v;
    u = (float*) malloc(R_C.rows * sizeof(float));
    size_t nbits_u = R_C.rows * sizeof(float);
    v = (float*) malloc(R_C.cols * sizeof(float));
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
    cudaStatus = cudaMalloc((void**) &dev_return, nThreadsPerBlock * nBlocks * sizeof(float));
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_return2, nThreadsPerBlock * nBlocks * sizeof(float));
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


    for (int oiter = 1; oiter <= maxiter; ++oiter) {
        float rankfundec = 0;
        float fundec_max = 0;
        int early_stop = 0;
        for (int t = 0; t < k; ++t) {
            if (early_stop >= 5) { break; }
            float* Wt = &W[t][0], * Ht = &H[t][0];

            cudaStatus = cudaMemcpy(dev_Wt_vec_t, Wt, nbits_u, cudaMemcpyHostToDevice);
            gpuErrchk(cudaStatus);
            cudaStatus = cudaMemcpy(dev_Ht_vec_t, Ht, nbits_v, cudaMemcpyHostToDevice);
            gpuErrchk(cudaStatus);
            if (oiter > 1) {
                UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx, dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, true, Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, true);
                cudaStatus = cudaDeviceSynchronize();
                gpuErrchk(cudaStatus);

            }

            float innerfundec_cur = 0, innerfundec_max = 0;
            int maxit = inneriter;
            for (int iter = 1; iter <= maxit; ++iter) {
                innerfundec_cur = 0;

                RankOneUpdate_DUAL_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx, dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, lambda, dev_return, parameters.do_nmf, Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, dev_return2);

                cudaStatus = cudaDeviceSynchronize();
                gpuErrchk(cudaStatus);
                cudaStatus = cudaMemcpy(Hostreduction, dev_return, nBlocks * nThreadsPerBlock * sizeof(float), cudaMemcpyDeviceToHost);
                gpuErrchk(cudaStatus);
                cudaStatus = cudaMemcpy(Hostreduction2, dev_return2, nBlocks * nThreadsPerBlock * sizeof(float), cudaMemcpyDeviceToHost);
                gpuErrchk(cudaStatus);

                for (size_t index = 0; index < nThreadsPerBlock * nBlocks; index++) {
                    innerfundec_cur += Hostreduction[index];
                }

                for (size_t index = 0; index < nThreadsPerBlock * nBlocks; index++) {
                    innerfundec_cur += Hostreduction2[index];
                }

                num_updates += R_C.cols;

                num_updates += Rt.cols;
                if ((innerfundec_cur < fundec_max * eps)) {
                    if (iter == 1) { early_stop += 1; }
                    break;
                }
                rankfundec += innerfundec_cur;
                innerfundec_max = std::max(innerfundec_max, innerfundec_cur);
                if (!(oiter == 1 && t == 0 && iter == 1)) {
                    fundec_max = std::max(fundec_max, innerfundec_cur);
                }
            }

            cudaStatus = cudaMemcpy(Wt, dev_Wt_vec_t, nbits_u, cudaMemcpyDeviceToHost);
            gpuErrchk(cudaStatus);
            cudaStatus = cudaMemcpy(Ht, dev_Ht_vec_t, nbits_v, cudaMemcpyDeviceToHost);
            gpuErrchk(cudaStatus);

            UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx, dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, false, Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, false);

            cudaStatus = cudaDeviceSynchronize();
            gpuErrchk(cudaStatus);
        }
    }

    free(u);
    free(v);
    free(Hostreduction);
    free(Hostreduction2);

    cudaFree(dev_Rcol_ptr);
    cudaFree(dev_Rrow_idx);
    cudaFree(dev_Rcol_ptr_T);
    cudaFree(dev_Rrow_idx_T);
    cudaFree(dev_Rval);
    cudaFree(dev_Rval_t);
    cudaFree(dev_Wt_vec_t);
    cudaFree(dev_Ht_vec_t);
    cudaFree(dev_return);
    return cudaStatus;
}

