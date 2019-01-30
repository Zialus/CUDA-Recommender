#include "pmf.h"
#include "tools.h"
#include "main.h"
#include "CCD_CUDA.h"

#define kind dynamic,500

inline float RankOneUpdate_Original_float(const smat_t& R, const long j, const vec_t& u, const float lambda, int do_nmf) {
    float g = 0, h = lambda;
    if (R.col_ptr[j + 1] == R.col_ptr[j]) { return 0; }
    for (long idx = R.col_ptr[j]; idx < R.col_ptr[j + 1]; ++idx) {
        int i = R.row_idx[idx];
        g += u[i] * R.val[idx];
        h += u[i] * u[i];
    }
    float newvj = g / h;
    if (do_nmf > 0 && newvj < 0) {
        newvj = 0;
    }
    return newvj;
}


inline float UpdateRating_Original_float(smat_t& R, const vec_t& Wt, const vec_t& Ht, bool add) {
    float loss = 0;
    if (add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
        for (int c = 0; c < R.cols; ++c) {
            float Htc = Ht[c], loss_inner = 0;
            for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx) {
                R.val[idx] += Wt[R.row_idx[idx]] * Htc;
                loss_inner += R.val[idx] * R.val[idx];
            }
            loss += loss_inner;
        }
        return loss;
    } else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
        for (int c = 0; c < R.cols; ++c) {
            float Htc = Ht[c], loss_inner = 0;
            for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx) {
                R.val[idx] -= Wt[R.row_idx[idx]] * Htc;
                loss_inner += R.val[idx] * R.val[idx];
            }
            loss += loss_inner;
        }
        return loss;
    }
}


void ccdr1(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param) {

    if (param.enable_cuda) {
        printf("CUDA enabled version.\n");
        kernel_wrapper_ccdpp_NV(R, T, W, H, param);
    } else {
        ccdr1_OMP(R, W, H, T, param);
    }
}

void ccdr1_OMP(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param) {
    int k = param.k;
    float lambda = param.lambda;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    // Create transpose view of R
    smat_t Rt;
    Rt = R.transpose();

    // H is a zero matrix now.
    for (int t = 0; t < k; ++t) { for (long c = 0; c < R.cols; ++c) { H[t][c] = 0; }}

    vec_t oldWt(R.rows), oldHt(R.cols);
    vec_t u(R.rows), v(R.cols);
    for (int oiter = 1; oiter <= param.maxiter; ++oiter) {

        double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0;

        for (int t = 0; t < k; ++t) {

            start = omp_get_wtime();
            vec_t& Wt = W[t], & Ht = H[t];
#pragma omp parallel for
            for (int i = 0; i < R.rows; ++i) {
                u[i] = Wt[i];
                oldWt[i] = u[i];
            }
#pragma omp parallel for
            for (int i = 0; i < R.cols; ++i) {
                v[i] = Ht[i];
                oldHt[i] = v[i];
            }

            // Create Rhat = R - Wt Ht^T
            if (oiter > 1) {
                UpdateRating_Original_float(R, Wt, Ht, true);
                UpdateRating_Original_float(Rt, Ht, Wt, true);
            }
            Itime += omp_get_wtime() - start;

            for (int iter = 1; iter <= param.maxinneriter; ++iter) {
                // Update H[t]
                start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < R.cols; ++c) {
                    v[c] = RankOneUpdate_Original_float(R, c, u, lambda * (R.col_ptr[c + 1] - R.col_ptr[c]), param.do_nmf);
                }
                Htime += omp_get_wtime() - start;

                // Update W[t]
                start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < Rt.cols; ++c) {
                    u[c] = RankOneUpdate_Original_float(Rt, c, v, lambda * (Rt.col_ptr[c + 1] - Rt.col_ptr[c]), param.do_nmf);
                }
                Wtime += omp_get_wtime() - start;
            }

            // Update R and Rt
            start = omp_get_wtime();

#pragma omp parallel for
            for (int i = 0; i < R.rows; ++i) { Wt[i] = u[i]; }
#pragma omp parallel for
            for (int i = 0; i < R.cols; ++i) { Ht[i] = v[i]; }

            UpdateRating_Original_float(R, u, v, false);
            UpdateRating_Original_float(Rt, v, u, false);

            Rtime += omp_get_wtime() - start;

            if (param.verbose) {
                printf("iter %d rank %d time %f", oiter, t + 1, Itime + Htime + Wtime + Rtime);
            }
            if (param.do_predict) {
                printf(" rmse %.10g", calrmse_r1(T, Wt, Ht, oldWt, oldHt));
            }
            if (param.verbose) { puts(""); }
            fflush(stdout);
        }
        calculate_rmse_directly(W, H, T, oiter, param.k, false);
        double rmse = calrmse(T, W, H, false, true);
        printf("Test RMSE = %f , iteration number %d\n", rmse, oiter);
    }
    omp_set_num_threads(num_threads_old);
}