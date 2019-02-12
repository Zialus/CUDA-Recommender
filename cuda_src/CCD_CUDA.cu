#include "CCD_CUDA.h"

__global__ void RankOneUpdate_v_kernel(const long Rcols,
                                       const long* Rcol_ptr,
                                       const long* Rrow_idx,
                                       const float* Rval,

                                       float* u,
                                       float* v,
                                       const float lambda,
                                       const int do_nmf
) {
    long thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    long total_threads = blockDim.x * gridDim.x;

    for (long c = thread_id; c < Rcols; c += total_threads) {
        v[c] = RankOneUpdate_dev(Rcol_ptr, Rrow_idx, Rval, c, u,
                                 lambda * (Rcol_ptr[c + 1] - Rcol_ptr[c]), do_nmf);
    }

}

__global__ void RankOneUpdate_u_kernel(const long Rcols_t,
                                       const long* Rcol_ptr_t,
                                       const long* Rrow_idx_t,
                                       const float* Rval_t,

                                       float* u,
                                       float* v,
                                       const float lambda,
                                       const int do_nmf
) {
    long thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    long total_threads = blockDim.x * gridDim.x;

    for (long c = thread_id; c < Rcols_t; c += total_threads) {
        u[c] = RankOneUpdate_dev(Rcol_ptr_t, Rrow_idx_t, Rval_t, c, v,
                                 lambda * (Rcol_ptr_t[c + 1] - Rcol_ptr_t[c]), do_nmf);
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
    long thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    long total_threads = blockDim.x * gridDim.x;

    for (long i = thread_id; i < Rcols; i += total_threads) {
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

    for (long i = thread_id; i < Rcols_t; i += total_threads) {
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

__global__ void UpdateRating_W_kernel(const long Rcols,
                                      const long* Rcol_ptr,
                                      const long* Rrow_idx,
                                      float* Rval,

                                      const float* Wt_vec_t,
                                      const float* Ht_vec_t,
                                      const bool add


) {
    long thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    long total_threads = blockDim.x * gridDim.x;

    for (long i = thread_id; i < Rcols; i += total_threads) {
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

}

__global__ void UpdateRating_H_kernel(const long Rcols_t,
                                      const long* Rcol_ptr_t,
                                      const long* Rrow_idx_t,
                                      float* Rval_t,

                                      const float* Wt_vec_t,
                                      const float* Ht_vec_t,
                                      const bool add_t

) {
    long thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    long total_threads = blockDim.x * gridDim.x;

    for (long i = thread_id; i < Rcols_t; i += total_threads) {
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
    // Reset GPU.
    cudaStatus = cudaDeviceReset();
    gpuErrchk(cudaStatus);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    gpuErrchk(cudaStatus);

    cudaStatus = ccdpp_NV(R, T, W, H, parameters);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CCD FAILED: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceReset();
    gpuErrchk(cudaStatus);
}

inline cudaError_t
updateRating(unsigned int nThreadsPerBlock, unsigned int nBlocks, const smat_t& R_C, const smat_t& Rt,
             const long* dev_Rcol_ptr, const long* dev_Rrow_idx, const long* dev_Rcol_ptr_T, const long* dev_Rrow_idx_T,
             float* dev_Rval, float* dev_Rval_t, const float* dev_Wt_vec_t, const float* dev_Ht_vec_t, const bool add,
             cudaError_t& cudaStatus) {

    UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx,
                    dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, add, Rt.cols,
                    dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, add);

//    UpdateRating_W_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx,
//            dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, add);
//    UpdateRating_H_kernel<<<nBlocks, nThreadsPerBlock>>>(Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T,
//            dev_Rval_t, dev_Wt_vec_t, dev_Ht_vec_t, add);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    gpuErrchk(cudaStatus);
    cudaStatus = cudaDeviceSynchronize();
    gpuErrchk(cudaStatus);
    return cudaStatus;
}

inline cudaError_t
RankOneUpdate(const parameter& parameters, unsigned int nThreadsPerBlock, unsigned int nBlocks, float lambda,
              const smat_t& R_C, const smat_t& Rt, const long* dev_Rcol_ptr, const long* dev_Rrow_idx,
              const long* dev_Rcol_ptr_T, const long* dev_Rrow_idx_T, const float* dev_Rval,
              const float* dev_Rval_t, float* dev_Wt_vec_t,float* dev_Ht_vec_t, cudaError_t& cudaStatus) {

    RankOneUpdate_v_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx,
            dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, lambda, parameters.do_nmf);
    RankOneUpdate_u_kernel<<<nBlocks, nThreadsPerBlock>>>(Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T,
            dev_Rval_t, dev_Wt_vec_t, dev_Ht_vec_t, lambda, parameters.do_nmf);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    gpuErrchk(cudaStatus);
    cudaStatus = cudaDeviceSynchronize();
    gpuErrchk(cudaStatus);
    return cudaStatus;
}


cudaError_t ccdpp_NV(smat_t& R_C, testset_t& T, mat_t& W, mat_t& H, parameter& parameters) {
    unsigned nThreadsPerBlock = parameters.nThreadsPerBlock;
    unsigned nBlocks = parameters.nBlocks;

    cudaError_t cudaStatus;

    int k = parameters.k;
    float lambda = parameters.lambda;

    // Create transpose view of R
    smat_t Rt;
    Rt = R_C.transpose();

    long* dev_Rcol_ptr = nullptr;
    long* dev_Rrow_idx = nullptr;
    long* dev_Rcol_ptr_T = nullptr;
    long* dev_Rrow_idx_T = nullptr;
    float* dev_Rval = nullptr;
    float* dev_Rval_t = nullptr;

    float* dev_Wt_vec_t = nullptr; //u
    float* dev_Ht_vec_t = nullptr; //v

    float* dev_W_ = nullptr;
    float* dev_H_ = nullptr;

    size_t nbits_W_ = R_C.rows * k * sizeof(float);
    float* W_ = (float*) malloc(nbits_W_);
    size_t nbits_H_ = R_C.cols * k * sizeof(float);
    float* H_ = (float*) malloc(nbits_H_);

    int indexPosition = 0;
    for (long i = 0; i < k; ++i) {
        for (int j = 0; j < R_C.rows; ++j) {
            W_[indexPosition] = W[i][j];
            ++indexPosition;
        }
    }

//    indexPosition = 0;
//    for (long i = 0; i < k; ++i) {
//        for (int j = 0; j < R_C.cols; ++j) {
//            H_[indexPosition] = H[i][j];
//            ++indexPosition;
//        }
//    }

    cudaStatus = cudaMalloc((void**) &dev_W_, nbits_W_);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_H_, nbits_H_);
    gpuErrchk(cudaStatus);

    cudaStatus = cudaMemcpy(dev_W_, W_, nbits_W_, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemset(dev_H_, 0, nbits_H_);
//    cudaStatus = cudaMemcpy(dev_H_, H_, nbits_H_, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);


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

    cudaStatus = cudaMemcpy(dev_Rcol_ptr, R_C.col_ptr, R_C.nbits_col_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rrow_idx, R_C.row_idx, R_C.nbits_row_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rcol_ptr_T, Rt.col_ptr, Rt.nbits_col_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rrow_idx_T, Rt.row_idx, Rt.nbits_row_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rval, R_C.val, R_C.nbits_val, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_Rval_t, Rt.val, Rt.nbits_val, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);

    float* rmse = (float*) malloc((T.nnz) * sizeof(float));

    long* d_test_row;
    long* d_test_col;
    float* d_test_val;
    float* d_pred_v;
    float* d_rmse;

    gpuErrchk(cudaMalloc((void**) &d_test_row, (T.nnz + 1) * sizeof(long)));
    gpuErrchk(cudaMalloc((void**) &d_test_col, (T.nnz + 1) * sizeof(long)));
    gpuErrchk(cudaMalloc((void**) &d_test_val, (T.nnz + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &d_pred_v, (T.nnz + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &d_rmse, (T.nnz + 1) * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_test_row, T.test_row, (T.nnz + 1) * sizeof(long), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_test_col, T.test_col, (T.nnz + 1) * sizeof(long), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_test_val, T.test_val, (T.nnz + 1) * sizeof(float), cudaMemcpyHostToDevice));

    float update_time_acc = 0;
    float rank_time_acc = 0;

    for (int oiter = 1; oiter <= parameters.maxiter; ++oiter) {

        float update_time = 0;
        float rank_time = 0;
        GpuTimer update_timer;
        GpuTimer rmse_timer;
        GpuTimer rank_timer;

        for (int t = 0; t < k; ++t) {

            dev_Wt_vec_t = dev_W_ + t * R_C.rows; //u
            dev_Ht_vec_t = dev_H_ + t * R_C.cols; //v

            if (oiter > 1) {
                update_timer.Start();
                cudaStatus = updateRating(nThreadsPerBlock, nBlocks, R_C,
                                          Rt, dev_Rcol_ptr, dev_Rrow_idx, dev_Rcol_ptr_T, dev_Rrow_idx_T,
                                          dev_Rval, dev_Rval_t, dev_Wt_vec_t, dev_Ht_vec_t, true, cudaStatus);

                update_timer.Stop();
                update_time += update_timer.Elapsed();
            }

            rank_timer.Start();
            for (int iter = 1; iter <= parameters.maxinneriter; ++iter) {
                cudaStatus = RankOneUpdate(parameters, nThreadsPerBlock, nBlocks, lambda, R_C, Rt, dev_Rcol_ptr,
                                           dev_Rrow_idx, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval,
                                           dev_Rval_t, dev_Wt_vec_t, dev_Ht_vec_t, cudaStatus);
            }
            rank_timer.Stop();
            rank_time += rank_timer.Elapsed();

            update_timer.Start();
            cudaStatus = updateRating(nThreadsPerBlock, nBlocks, R_C,
                                      Rt, dev_Rcol_ptr, dev_Rrow_idx, dev_Rcol_ptr_T, dev_Rrow_idx_T,
                                      dev_Rval, dev_Rval_t, dev_Wt_vec_t, dev_Ht_vec_t, false, cudaStatus);

            update_timer.Stop();
            update_time += update_timer.Elapsed();
        }

        update_time_acc += update_time;
        rank_time_acc += rank_time;
        /*********************Check RMSE*********************/
        rmse_timer.Start();

        gpuErrchk(cudaMemset(d_rmse, 0, (T.nnz + 1) * sizeof(float)));
        gpuErrchk(cudaMemset(d_pred_v, 0, (T.nnz + 1) * sizeof(float)));
        GPU_rmse<<<(T.nnz + 1023) / 1024, 1024>>>(d_test_row, d_test_col, d_test_val, d_pred_v, d_rmse,
                dev_W_, dev_H_, T.nnz, k, R_C.rows, R_C.cols, false);
        cudaStatus = cudaGetLastError();
        gpuErrchk(cudaStatus);
        cudaStatus = cudaDeviceSynchronize();
        gpuErrchk(cudaStatus);

        double tot_rmse = 0;
        double f_rmse = 0;
        gpuErrchk(cudaMemcpy(rmse, d_rmse, (T.nnz + 1) * sizeof(float), cudaMemcpyDeviceToHost));

        for (long i = 0; i < T.nnz; ++i) {
            tot_rmse += rmse[i];
        }
        f_rmse = sqrtf(tot_rmse / T.nnz);
        rmse_timer.Stop();

        float rmse_time = rmse_timer.Elapsed();
        printf("[-INFO-] iteration num %d \trank_time %.4lf|%.4lf s \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n",
                oiter, rank_time, rank_time_acc, update_time, update_time_acc, f_rmse, rmse_time);
    }

    cudaStatus = cudaMemcpy(H_, dev_H_, nbits_H_, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(W_, dev_W_, nbits_W_, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaStatus);

    indexPosition = 0;
    for (long i = 0; i < k; ++i) {
        for (int j = 0; j < R_C.rows; ++j) {
            W[i][j] = W_[indexPosition];
            ++indexPosition;
        }
    }
    indexPosition = 0;
    for (long i = 0; i < k; ++i) {
        for (int j = 0; j < R_C.cols; ++j) {
            H[i][j] = H_[indexPosition];
            ++indexPosition;
        }
    }

    free(W_);
    free(H_);

    cudaFree(dev_W_);
    cudaFree(dev_H_);

    cudaFree(dev_Rcol_ptr);
    cudaFree(dev_Rrow_idx);
    cudaFree(dev_Rcol_ptr_T);
    cudaFree(dev_Rrow_idx_T);
    cudaFree(dev_Rval);
    cudaFree(dev_Rval_t);

    return cudaStatus;
}
