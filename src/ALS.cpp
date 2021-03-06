#include "extras.h"
#include "ALS.h"

#define kind dynamic,500

void choldc1(int n, float** a, float* p) {
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

void choldcsl(int n, float** A) {
    float* p = (float*) malloc(n * sizeof(float));
    choldc1(n, A, p);
    for (int i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (int j = i + 1; j < n; ++j) {
            double sum = 0;
            for (int k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = (float) sum / p[j];
        }
    }
    free(p);
}

void inverseMatrix_CholeskyMethod(int n, float** A) {
    choldcsl(n, A);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        A[i][i] *= A[i][i];
        for (int k = i + 1; k < n; ++k) {
            A[i][i] += A[k][i] * A[k][i];
        }
        for (int j = i + 1; j < n; ++j) {
            for (int k = j; k < n; ++k) {
                A[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }
}

void Mt_byM_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (int I = 0; I < j; ++I) {
        for (int J = I; J < j; ++J) {
            SUM = 0.0f;
            for (int K = 0; K < i; ++K) {
                //printf("%.3f %.3f\n", M[K][I], M[K][J]);
                SUM += M[K][I] * M[K][J];
            }
            Result[J][I] = SUM;
            Result[I][J] = SUM;
        }
    }
}

void ALS_OMP(SparseMatrix& R, MatData& W, MatData& H, TestData& T, parameter& param) {
    int k = param.k;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    // Create transpose view of R
    SparseMatrix Rt;
    Rt = R.get_shallow_transpose();

    double update_time_acc = 0;

    for (int iter = 0; iter < param.maxiter; ++iter) {

        double start = omp_get_wtime();

        //optimize W over H
#pragma omp parallel for schedule(kind)
        for (int Rw = 0; Rw < R.rows; ++Rw) {
            float* Wr = &W[Rw][0];
            int omegaSize = R.get_csr_row_ptr()[Rw + 1] - R.get_csr_row_ptr()[Rw];
            float** subMatrix;

            if (omegaSize > 0) {
                float* subVector = (float*) malloc(k * sizeof(float));
                subMatrix = (float**) malloc(k * sizeof(float*));
                for (int i = 0; i < k; ++i) {
                    subMatrix[i] = (float*) malloc(k * sizeof(float));
                }

                //a trick to avoid malloc
                float** H_Omega = (float**) malloc(omegaSize * sizeof(float*));
                for (unsigned idx = R.get_csr_row_ptr()[Rw], i=0; idx < R.get_csr_row_ptr()[Rw + 1]; ++idx, ++i) {
                    H_Omega[i] = &H[R.get_csr_col_indx()[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, H_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (int c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + param.lambda;
                }

                //invert sub-matrix
                inverseMatrix_CholeskyMethod(k, subMatrix);


                //sparse multiplication
                for (int c = 0; c < k; ++c) {
                    subVector[c] = 0;
                    for (unsigned idx = R.get_csr_row_ptr()[Rw]; idx < R.get_csr_row_ptr()[Rw + 1]; ++idx) {
                        subVector[c] += Rt.get_csc_val()[idx] * H[R.get_csr_col_indx()[idx]][c];
                    }
                }

                //multiply subVector by subMatrix
                for (int c = 0; c < k; ++c) {
                    Wr[c] = 0;
                    for (int subVid = 0; subVid < k; ++subVid) {
                        Wr[c] += subVector[subVid] * subMatrix[c][subVid];
                    }
                }


                for (int i = 0; i < k; ++i) {
                    free(subMatrix[i]);
                }
                free(subMatrix);
                free(subVector);
                free(H_Omega);
            } else {
                for (int c = 0; c < k; ++c) {
                    Wr[c] = 0.0f;
                    //printf("%.3f ", Wr[c]);
                }
                //printf("\n");
            }
        }

        //optimize H over W
#pragma omp parallel for schedule(kind)
        for (int Rh = 0; Rh < R.cols; ++Rh) {
            float* Hr = &H[Rh][0];
            unsigned omegaSize = R.get_csc_col_ptr()[Rh + 1] - R.get_csc_col_ptr()[Rh];
            float** subMatrix;

            if (omegaSize > 0) {
                float* subVector = (float*) malloc(k * sizeof(float));
                subMatrix = (float**) malloc(k * sizeof(float*));
                for (int i = 0; i < k; ++i) {
                    subMatrix[i] = (float*) malloc(k * sizeof(float));
                }

                //a trick to avoid malloc
                float** W_Omega = (float**) malloc(omegaSize * sizeof(float*));
                for (long idx = R.get_csc_col_ptr()[Rh], i = 0; idx < R.get_csc_col_ptr()[Rh + 1]; ++idx, ++i) {
                    W_Omega[i] = &W[R.get_csc_row_indx()[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, W_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (int c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + param.lambda;
                }

                //invert sub-matrix
                inverseMatrix_CholeskyMethod(k, subMatrix);


                //sparse multiplication
                for (int c = 0; c < k; ++c) {
                    subVector[c] = 0;
                    for (long idx = R.get_csc_col_ptr()[Rh]; idx < R.get_csc_col_ptr()[Rh + 1]; ++idx) {
                        subVector[c] += R.get_csc_val()[idx] * W[R.get_csc_row_indx()[idx]][c];
                    }
                }

                //multiply subVector by subMatrix
                for (int c = 0; c < k; ++c) {
                    Hr[c] = 0;
                    for (int subVid = 0; subVid < k; ++subVid) {
                        Hr[c] += subVector[subVid] * subMatrix[c][subVid];
                    }
                }


                for (int i = 0; i < k; ++i) {
                    free(subMatrix[i]);
                }
                free(subMatrix);
                free(subVector);
                free(W_Omega);
            } else {
                for (int c = 0; c < k; ++c) {
                    Hr[c] = 0.0f;
                }
            }
        }
        double end = omp_get_wtime();
        double update_time = end - start;
        update_time_acc+=update_time;

        start = omp_get_wtime();
        double rmse = calrmse(T, W, H, true, true);
        end = omp_get_wtime();
        double rmse_timer = end - start;

        printf("[-INFO-] iteration num %d \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n", iter+1, update_time, update_time_acc, rmse, rmse_timer);

    }
    omp_set_num_threads(num_threads_old);
}
