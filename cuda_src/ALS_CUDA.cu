#include "ALS_CUDA.h"

__device__ void choldc1_k(int n, float* a, float* p) {
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float sum = a[i * n + j];
            for (int k = i - 1; k >= 0; --k) {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrtf(sum);
            } else {
                a[j * n + i] = sum / p[i];
            }
        }
    }
}

__device__ void choldcsl_k(int n, float* A) {
    float* p = (float*) malloc(n * sizeof(float));
    choldc1_k(n, A, p);
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = 1 / p[i];
        for (int j = i + 1; j < n; ++j) {
            float sum = 0;
            for (int k = i; k < j; ++k) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            A[j * n + i] = sum / p[j];
        }
    }
    free(p);
}

__device__ void inverseMatrix_CholeskyMethod_k(int n, float* A) {
    int i, j, k;
    choldcsl_k(n, A);
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        A[i * n + i] *= A[i * n + i];
        for (k = i + 1; k < n; ++k) {
            A[i * n + i] += A[k * n + i] * A[k * n + i];
        }
        for (j = i + 1; j < n; ++j) {
            for (k = j; k < n; ++k) {
                A[i * n + j] += A[k * n + i] * A[k * n + j];
            }
        }
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
            A[i * n + j] = A[j * n + i];
        }
    }
}

//Multiply matrix M transpose by M 
__device__ void Mt_byM_multiply_k(long i, long j, float* H, float* Result, const long ptr, const unsigned* idx) {
    float SUM;
    for (int I = 0; I < j; ++I) {
        for (int J = I; J < j; ++J) {
            SUM = 0.0f;
            for (int K = 0; K < i; ++K) {
                unsigned offset = idx[ptr + K] * j;
//                printf("%.3f %.3f\n", H[offset + I], H[offset + J]);
                SUM += H[offset + I] * H[offset + J];
            }
            Result[J * j + I] = SUM;
            Result[I * j + J] = SUM;
        }
    }
}

__global__ void updateW_overH_kernel(const long rows, const unsigned* row_ptr, const unsigned* col_idx,
                                     const float* val_t, const float lambda, const unsigned k,
                                     float* W, float* H) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int Rw = tid; Rw < rows; Rw += blockDim.x * gridDim.x) {
        int offset_W = Rw * k;

        float* Wr = &W[offset_W];
        unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];

        if (omegaSize > 0) {
            float* subVector = (float*) malloc(k * sizeof(float));
            float* subMatrix = (float*) malloc(k * k * sizeof(float));

            Mt_byM_multiply_k(omegaSize, k, H, subMatrix, row_ptr[Rw], col_idx);

            //add lambda to diag of sub-matrix
            for (unsigned c = 0; c < k; c++) {
                subMatrix[c * k + c] += lambda;
            }

            //invert sub-matrix
            inverseMatrix_CholeskyMethod_k(k, subMatrix);

            //sparse multiplication
            for (unsigned c = 0; c < k; ++c) {
                subVector[c] = 0;
                for (unsigned idx = row_ptr[Rw]; idx < row_ptr[Rw + 1]; ++idx) {
                    subVector[c] += val_t[idx] * H[(col_idx[idx] * k) + c];
                }
            }

            //multiply subVector by subMatrix
            for (unsigned c = 0; c < k; ++c) {
                Wr[c] = 0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Wr[c] += subVector[subVid] * subMatrix[c * k + subVid];
                }
            }

            free(subMatrix);
            free(subVector);
        } else {
            for (unsigned c = 0; c < k; ++c) {
                Wr[c] = 0.0f;
            }
        }
    }
}

__global__ void updateH_overW_kernel(const long cols, const unsigned* col_ptr, const unsigned* row_idx,
                                     const float* val, const float lambda, const unsigned k,
                                     float* W, float* H) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int Rh = tid; Rh < cols; Rh += blockDim.x * gridDim.x) {
        int offset_H = Rh * k;

        float* Hr = &H[offset_H];
        unsigned omegaSize = col_ptr[Rh + 1] - col_ptr[Rh];

        if (omegaSize > 0) {
            float* subVector = (float*) malloc(k * sizeof(float));
            float* subMatrix = (float*) malloc(k * k * sizeof(float));

            Mt_byM_multiply_k(omegaSize, k, W, subMatrix, col_ptr[Rh], row_idx);

            //add lambda to diag of sub-matrix
            for (unsigned c = 0; c < k; c++) {
                subMatrix[c * k + c] += lambda;
            }

            //invert sub-matrix
            inverseMatrix_CholeskyMethod_k(k, subMatrix);

            //sparse multiplication
            for (unsigned c = 0; c < k; ++c) {
                subVector[c] = 0;
                for (unsigned idx = col_ptr[Rh]; idx < col_ptr[Rh + 1]; ++idx) {
                    subVector[c] += val[idx] * W[(row_idx[idx] * k) + c];
                }
            }

            //multiply subVector by subMatrix
            for (unsigned c = 0; c < k; ++c) {
                Hr[c] = 0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Hr[c] += subVector[subVid] * subMatrix[c * k + subVid];
                }
            }

            free(subMatrix);
            free(subVector);
        } else {
            for (unsigned c = 0; c < k; ++c) {
                Hr[c] = 0.0f;
            }
        }
    }
}

void kernel_wrapper_als_NV(SparseMatrix& R, TestData& T, MatData& W, MatData& H, parameter& parameters) {
    cudaError_t cudaStatus;
    // Reset GPU.
    cudaStatus = cudaDeviceReset();
    gpuErrchk(cudaStatus);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    gpuErrchk(cudaStatus);

    cudaStatus = als_NV(R, T, W, H, parameters);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ALS FAILED: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceReset();
    gpuErrchk(cudaStatus);
}

cudaError_t als_NV(SparseMatrix& R_C, TestData& T, MatData& W, MatData& H, parameter& parameters) {
    int nThreadsPerBlock = parameters.nThreadsPerBlock;
    int nBlocks = parameters.nBlocks;

    cudaError_t cudaStatus;

    float lambda = parameters.lambda;
    int k = parameters.k;

    // Create transpose view of R
    SparseMatrix Rt;
    Rt = R_C.get_shallow_transpose();

    unsigned* dev_col_ptr = nullptr;
    unsigned* dev_row_ptr = nullptr;
    unsigned* dev_row_idx = nullptr;
    unsigned* dev_col_idx = nullptr;
    float* dev_val_t = nullptr;
    float* dev_val = nullptr;

    float* dev_W_ = nullptr;
    float* dev_H_ = nullptr;


    size_t nbits_W_ = R_C.rows * k * sizeof(float);
    float* W_ = (float*) malloc(nbits_W_);
    size_t nbits_H_ = R_C.cols * k * sizeof(float);
    float* H_ = (float*) malloc(nbits_H_);

    int indexPosition = 0;
    for (long i = 0; i < R_C.rows; ++i) {
        for (int j = 0; j < k; ++j) {
            W_[indexPosition] = W[i][j];
            ++indexPosition;
        }
    }

    indexPosition = 0;
    for (long i = 0; i < R_C.cols; ++i) {
        for (int j = 0; j < k; ++j) {
            H_[indexPosition] = H[i][j];
            ++indexPosition;
        }
    }

    size_t nbits_col_ptr = (R_C.cols + 1) * sizeof(unsigned);
    size_t nbits_row_ptr = (R_C.rows + 1) * sizeof(unsigned);

    size_t nbits_idx = R_C.nnz * sizeof(unsigned);

    size_t nbits_val = R_C.nnz * sizeof(DTYPE);


    cudaStatus = cudaMalloc((void**) &dev_W_, nbits_W_);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_H_, nbits_H_);
    gpuErrchk(cudaStatus);

    cudaStatus = cudaMemcpy(dev_W_, W_, nbits_W_, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_H_, H_, nbits_H_, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);


    cudaStatus = cudaMalloc((void**) &dev_col_ptr, nbits_col_ptr);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_row_ptr, nbits_row_ptr);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_row_idx, nbits_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_col_idx, nbits_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_val_t, nbits_val);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_val, nbits_val);
    gpuErrchk(cudaStatus);


    cudaStatus = cudaMemcpy(dev_col_ptr, R_C.get_csc_col_ptr(), nbits_col_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_row_ptr, R_C.get_csr_row_ptr(), nbits_row_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_row_idx, R_C.get_csc_row_indx(), nbits_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_col_idx, R_C.get_csr_col_indx(), nbits_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_val_t, Rt.get_csc_val(), nbits_val, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_val, R_C.get_csc_val(), nbits_val, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);


    float* rmse = (float*) malloc((T.nnz + 1) * sizeof(float));

    unsigned* d_test_row;
    unsigned* d_test_col;
    float* d_test_val;
    float* d_pred_v;
    float* d_rmse;

    gpuErrchk(cudaMalloc((void**) &d_test_row, (T.nnz + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc((void**) &d_test_col, (T.nnz + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc((void**) &d_test_val, (T.nnz + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &d_pred_v, (T.nnz + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &d_rmse, (T.nnz + 1) * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_test_row, T.getTestRow(), (T.nnz + 1) * sizeof(unsigned), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_test_col, T.getTestCol(), (T.nnz + 1) * sizeof(unsigned), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_test_val, T.getTestVal(), (T.nnz + 1) * sizeof(float), cudaMemcpyHostToDevice));

    float update_time_acc = 0;

    for (int iter = 1; iter <= parameters.maxiter; ++iter) {
        float update_time = 0;
        GpuTimer update_timer;
        GpuTimer rmse_timer;
        update_timer.Start();
        /********************optimize W over H***************/
        updateW_overH_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.rows, dev_row_ptr, dev_col_idx,
                dev_val_t, lambda, k, dev_W_, dev_H_);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        gpuErrchk(cudaStatus);
        cudaStatus = cudaDeviceSynchronize();
        gpuErrchk(cudaStatus);

        /*******************optimize H over W****************/
        updateH_overW_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_col_ptr, dev_row_idx,
                dev_val, lambda, k, dev_W_, dev_H_);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        gpuErrchk(cudaStatus);
        cudaStatus = cudaDeviceSynchronize();
        gpuErrchk(cudaStatus);
        update_timer.Stop();
        update_time = update_timer.Elapsed();
        update_time_acc += update_time;
        /*********************Check RMSE*********************/
        rmse_timer.Start();

        gpuErrchk(cudaMemset(d_rmse, 0, (T.nnz + 1) * sizeof(float)));
        gpuErrchk(cudaMemset(d_pred_v, 0, (T.nnz + 1) * sizeof(float)));
        GPU_rmse<<<(T.nnz + 1023) / 1024, 1024>>>(d_test_row, d_test_col, d_test_val, d_pred_v, d_rmse,
                dev_W_, dev_H_, T.nnz, k, R_C.rows, R_C.cols, true);
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
        printf("[-INFO-] iteration num %d \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n",
               iter, update_time, update_time_acc, f_rmse, rmse_time);
    }

    cudaStatus = cudaMemcpy(H_, dev_H_, nbits_H_, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(W_, dev_W_, nbits_W_, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaStatus);

    indexPosition = 0;
    for (long i = 0; i < R_C.rows; ++i) {
        for (int j = 0; j < k; ++j) {
            W[i][j] = W_[indexPosition];
            ++indexPosition;
        }
    }
    indexPosition = 0;
    for (long i = 0; i < R_C.cols; ++i) {
        for (int j = 0; j < k; ++j) {
            H[i][j] = H_[indexPosition];
            ++indexPosition;
        }
    }

    free(W_);
    free(H_);

    free(rmse);

    gpuErrchk(cudaFree(dev_W_));
    gpuErrchk(cudaFree(dev_H_));

    gpuErrchk(cudaFree(dev_col_ptr));
    gpuErrchk(cudaFree(dev_row_ptr));
    gpuErrchk(cudaFree(dev_row_idx));
    gpuErrchk(cudaFree(dev_col_idx));
    gpuErrchk(cudaFree(dev_val_t));
    gpuErrchk(cudaFree(dev_val));

    gpuErrchk(cudaFree(d_test_row));
    gpuErrchk(cudaFree(d_test_col));
    gpuErrchk(cudaFree(d_test_val));
    gpuErrchk(cudaFree(d_pred_v));
    gpuErrchk(cudaFree(d_rmse));

    return cudaStatus;
}
