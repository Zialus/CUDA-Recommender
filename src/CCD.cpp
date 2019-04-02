#include "extras.h"
#include "CCD.h"

#define kind dynamic,500

inline float RankOneUpdate_Original_float(const smat_t& R, const long j, const vec_t& u, const float lambda) {
    float g = 0, h = lambda;
    if (R.col_ptr[j + 1] == R.col_ptr[j]) { return 0; }
    for (long idx = R.col_ptr[j]; idx < R.col_ptr[j + 1]; ++idx) {
        long i = R.row_idx[idx];
        g += u[i] * R.val[idx];
        h += u[i] * u[i];
    }
    float newvj = g / h;
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

void ccdr1_OMP(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param) {
    float lambda = param.lambda;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    // Create transpose view of R
    smat_t Rt;
    Rt = R.transpose();

    // H is a zero matrix now.
    for (unsigned t = 0; t < param.k; ++t) {
        for (long c = 0; c < R.cols; ++c) {
            H[t][c] = 0;
        }
    }

    vec_t u(R.rows), oldWt(R.rows);
    vec_t v(R.cols), oldHt(R.cols);

    double update_time_acc = 0;
    double rank_time_acc = 0;

    for (int oiter = 1; oiter <= param.maxiter; ++oiter) {

//        double total_time = 0;
        double update_time = 0;
        double rank_time = 0;

        for (unsigned t = 0; t < param.k; ++t) {

            double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0;

            double start = omp_get_wtime();

            vec_t& Wt = W[t];
            vec_t& Ht = H[t];

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
                    v[c] = RankOneUpdate_Original_float(R, c, u, lambda * (R.col_ptr[c + 1] - R.col_ptr[c]));
                }
                Htime += omp_get_wtime() - start;

                // Update W[t]
                start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < Rt.cols; ++c) {
                    u[c] = RankOneUpdate_Original_float(Rt, c, v, lambda * (Rt.col_ptr[c + 1] - Rt.col_ptr[c]));
                }
                Wtime += omp_get_wtime() - start;
            }

            // Update R and Rt
            start = omp_get_wtime();

#pragma omp parallel for
            for (long i = 0; i < R.rows; ++i) { Wt[i] = u[i]; }
#pragma omp parallel for
            for (long i = 0; i < R.cols; ++i) { Ht[i] = v[i]; }

            UpdateRating_Original_float(R, u, v, false);
            UpdateRating_Original_float(Rt, v, u, false);

            Rtime += omp_get_wtime() - start;

            update_time += Rtime + Itime;
            rank_time += Wtime + Htime;

//            if (param.verbose) {
//                total_time = (Itime + Htime + Wtime + Rtime);
//                printf("iter %d rank %d time %f", oiter, t + 1, total_time);
//                if (param.do_predict) {
//                    printf(" rmse %f", calrmse_r1(T, Wt, Ht, oldWt, oldHt));
//                }
//                printf("\n");
//            }
        }

        update_time_acc += update_time;
        rank_time_acc += rank_time;

        double start = omp_get_wtime();
        double rmse = calrmse(T, W, H, false, true);
        double rmse_time = omp_get_wtime() - start;

        printf("[-INFO-] iteration num %d \trank_time %.4lf|%.4lf s \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n",
               oiter, rank_time, rank_time_acc, update_time, update_time_acc, rmse, rmse_time);

    }
    omp_set_num_threads(num_threads_old);
}
