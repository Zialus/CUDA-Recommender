#include "ALS_onCUDA.h"

// CUDA kernel to pause for at least num_cycle cycles
//__device__ void sleep(int64_t num_cycles) {
//    int64_t cycles = 0;
//    int64_t start = clock64();
//    while (cycles < num_cycles) {
//        cycles = clock64() - start;
//    }
//}

__device__ void choldc1_k(int n, float** a, float* p) {
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float sum = a[i][j];
            for (int k = i - 1; k >= 0; --k) {
                sum -= a[i][k] * a[j][k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrtf(sum);
            } else {
                a[j][i] = sum / p[i];
            }
        }
    }
}

__device__ void choldcsl_k(int n, float** A) {
    float* p = (float*) malloc(n * sizeof(float));
    choldc1_k(n, A, p);
    for (int i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (int j = i + 1; j < n; ++j) {
            float sum = 0;
            for (int k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / p[j];
        }
    }
    free(p);
}

__device__ void inverseMatrix_CholeskyMethod_k(int n, float** A) {
    int i, j, k;
    choldcsl_k(n, A);
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        A[i][i] *= A[i][i];
        for (k = i + 1; k < n; ++k) {
            A[i][i] += A[k][i] * A[k][i];
        }
        for (j = i + 1; j < n; ++j) {
            for (k = j; k < n; ++k) {
                A[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }
}

//Multiply matrix M transpose by M 
__device__ void Mt_byM_multiply_k(long i, long j, float* H, float** Result, const long ptr, const long* idx) {
    float SUM;
    for (int I = 0; I < j; ++I) {
        for (int J = I; J < j; ++J) {
            SUM = 0.0f;
            for (int K = 0; K < i; ++K) {
                unsigned offset = idx[ptr + K] * j;
                //printf("%.3f %.3f\n", M[K][I], M[K][J]);
                //printf("%.3f %.3f\n", H[( offset) + I], H[( offset) + J]);
                SUM += H[offset + I] * H[offset + J];
            }
            Result[J][I] = SUM;
            Result[I][J] = SUM;
        }
    }
}

__global__ void updateW_overH_kernel(const long rows, const long* row_ptr, const long* col_idx, const long* colMajored_sparse_idx, const float* val, const float lambda, const unsigned k, float* W, float* H) {
    assert(row_ptr);
    assert(colMajored_sparse_idx);
    assert(val);
    assert(W);
    assert(H);

//    int tid = blockIdx.x * blockDim.x + threadIdx.x;

//    if (tid == 0) {
//        printf("OLA 1\n");
//    }

    //optimize W over H
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    for (int Rw = ii; Rw < rows; Rw += blockDim.x * gridDim.x) {
        //int offset_W = Rw*k;
        //int offset_H = Rw*cols;

        float* Wr = &W[Rw * k];
        unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];
        float** subMatrix;
        float* subVector;

        if (omegaSize > 0) {
            subVector = (float*) malloc(k * sizeof(float));
            subMatrix = (float**) malloc(k * sizeof(float*));

//            if (tid == 0) {
//                printf("OLA 2\n");
//            }

            assert(subVector);
            assert(subMatrix);
            for (unsigned i = 0; i < k; ++i) {
//                if (tid == 0) {
//                    printf("OLA 3.1,i=%d\n", i);
//                } //else { sleep(1000000000);}
                subMatrix[i] = (float*) malloc(k * sizeof(float));
//                if (tid == 0) {
//                    printf("OLA 3.2,i=%d\n", i);
//                }
                assert(subMatrix);
            }

//            if (tid == 0) {
//                printf("OLA 4\n");
//            }

            Mt_byM_multiply_k(omegaSize, k, H, subMatrix, row_ptr[Rw], col_idx);

            //add lambda to diag of sub-matrix
            for (unsigned c = 0; c < k; c++) {
                subMatrix[c][c] = subMatrix[c][c] + lambda;
            }

            //invert sub-matrix
            inverseMatrix_CholeskyMethod_k(k, subMatrix);


            //sparse multiplication
            for (unsigned c = 0; c < k; ++c) {
                subVector[c] = 0;
                for (long idx = row_ptr[Rw]; idx < row_ptr[Rw + 1]; ++idx) {
                    unsigned idx2 = colMajored_sparse_idx[idx];
                    subVector[c] += val[idx2] * H[(col_idx[idx] * k) + c];
                }
            }

            //multiply subVector by subMatrix
            for (unsigned c = 0; c < k; ++c) {
                Wr[c] = 0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Wr[c] += subVector[subVid] * subMatrix[c][subVid];
                }
            }


            for (unsigned i = 0; i < k; ++i) {
                free(subMatrix[i]);
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

__global__ void updateH_overW_kernel(const long cols, const long* col_ptr, const long* row_idx, const float* val, const float lambda, const unsigned k, float* W, float* H) {
    //optimize H over W
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    for (int Rh = ii; Rh < cols; Rh += blockDim.x * gridDim.x) {
        float* Hr = &H[Rh * k];
        //int offset_H = Rh*k;
        unsigned omegaSize = col_ptr[Rh + 1] - col_ptr[Rh];
        float** subMatrix;// ** W_Omega;
        float* subVector;

        if (omegaSize > 0) {
            subVector = (float*) malloc(k * sizeof(float));
            subMatrix = (float**) malloc(k * sizeof(float*));
            for (unsigned i = 0; i < k; ++i) {
                subMatrix[i] = (float*) malloc(k * sizeof(float));
            }

            Mt_byM_multiply_k(omegaSize, k, W, subMatrix, col_ptr[Rh], row_idx);

            //add lambda to diag of sub-matrix
            for (unsigned c = 0; c < k; c++) {
                subMatrix[c][c] = subMatrix[c][c] + lambda;
            }

            //invert sub-matrix
            inverseMatrix_CholeskyMethod_k(k, subMatrix);


            //sparse multiplication
            for (unsigned c = 0; c < k; ++c) {
                subVector[c] = 0;
                for (long idx = col_ptr[Rh]; idx < col_ptr[Rh + 1]; ++idx) {
                    subVector[c] += val[idx] * W[(row_idx[idx] * k) + c];
                }
            }

            //multiply subVector by subMatrix
            for (unsigned c = 0; c < k; ++c) {
                Hr[c] = 0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Hr[c] += subVector[subVid] * subMatrix[c][subVid];
                }
            }


            for (unsigned i = 0; i < k; ++i) {
                free(subMatrix[i]);
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

void kernel_wrapper_als_NV(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters) {
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

cudaError_t als_NV(smat_t& R_C, testset_t& T, mat_t& W, mat_t& H, parameter& parameters) {
    int nThreadsPerBlock = parameters.nThreadsPerBlock;
    int nBlocks = parameters.nBlocks;

    cudaError_t cudaStatus;

    float lambda = parameters.lambda;
    int k = parameters.k;

    long* dev_col_ptr = nullptr;
    long* dev_row_ptr = nullptr;
    long* dev_row_idx = nullptr;
    long* dev_col_idx = nullptr;
    long* dev_colMajored_sparse_idx = nullptr;
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

    cudaStatus = cudaMalloc((void**) &dev_W_, nbits_W_);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_H_, nbits_H_);
    gpuErrchk(cudaStatus);

    cudaStatus = cudaMemcpy(dev_W_, W_, nbits_W_, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_H_, H_, nbits_H_, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);


    cudaStatus = cudaMalloc((void**) &dev_col_ptr, R_C.nbits_col_ptr);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_row_ptr, R_C.nbits_row_ptr);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_row_idx, R_C.nbits_row_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_col_idx, R_C.nbits_col_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_colMajored_sparse_idx, R_C.nbits_colMajored_sparse_idx);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMalloc((void**) &dev_val, R_C.nbits_val);
    gpuErrchk(cudaStatus);


    cudaStatus = cudaMemcpy(dev_col_ptr, R_C.col_ptr, R_C.nbits_col_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_row_ptr, R_C.row_ptr, R_C.nbits_row_ptr, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_row_idx, R_C.row_idx, R_C.nbits_row_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_col_idx, R_C.col_idx, R_C.nbits_col_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_colMajored_sparse_idx, R_C.colMajored_sparse_idx, R_C.nbits_colMajored_sparse_idx, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);
    cudaStatus = cudaMemcpy(dev_val, R_C.val, R_C.nbits_val, cudaMemcpyHostToDevice);
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

    for (int iter = 1; iter <= parameters.maxiter; ++iter) {

        GpuTimer t;
        t.Start();
        /********************optimize W over H***************/
        updateW_overH_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.rows, dev_row_ptr, dev_col_idx,
                dev_colMajored_sparse_idx, dev_val, lambda, k, dev_W_, dev_H_);
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
        t.Stop();
        /*********************Check RMSE*********************/
        gpuErrchk(cudaMemset(d_rmse, 0, (T.nnz + 1) * sizeof(float)));
        gpuErrchk(cudaMemset(d_pred_v, 0, (T.nnz + 1) * sizeof(float)));
        GPU_rmse<<<(T.nnz + 1023) / 1024, 1024>>>(d_test_row, d_test_col, d_test_val, d_pred_v, d_rmse,
                dev_W_, dev_H_, T.nnz, k, R_C.rows, R_C.cols, true);
        cudaStatus = cudaGetLastError();
        gpuErrchk(cudaStatus);
        cudaStatus = cudaDeviceSynchronize();
        gpuErrchk(cudaStatus);

        float tot_rmse = 0;
        float f_rmse = 0;
        gpuErrchk(cudaMemcpy(rmse, d_rmse, (T.nnz + 1) * sizeof(float), cudaMemcpyDeviceToHost));

        for (unsigned i = 0; i < T.nnz; ++i) {
            tot_rmse += rmse[i];
        }
        f_rmse = sqrtf(tot_rmse / T.nnz);
        printf("iter %d RMSE %f time %f\n", iter, f_rmse, t.Elapsed());
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

    cudaFree(dev_W_);
    cudaFree(dev_H_);

    cudaFree(dev_col_ptr);
    cudaFree(dev_row_ptr);
    cudaFree(dev_row_idx);
    cudaFree(dev_col_idx);
    cudaFree(dev_colMajored_sparse_idx);
    cudaFree(dev_val);

    return cudaStatus;
}
