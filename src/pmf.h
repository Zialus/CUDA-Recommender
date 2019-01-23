#ifndef _PMF_H_
#define _PMF_H_

#include "util.h"

enum class solvertype {CCD, ALS};
enum { BOLDDRIVER, EXPDECAY };


class parameter {
public:
    solvertype solver_type;
    int k;
    int threads;
    int maxiter, maxinneriter;
    float lambda;
    float rho;
    float eps;   // for the fundec stop-cond in ccdr1
    int do_predict, verbose;
    int do_nmf;  // non-negative matrix factorization
    bool enable_cuda;
    int nBlocks;
    int nThreadsPerBlock;

    parameter() {
        solver_type = solvertype::CCD;
        k = 10;
        rho = 1e-3f;
        maxiter = 5;
        maxinneriter = 5;
        lambda = 0.1f;
        threads = 4;
        eps = 1e-3f;
        do_predict = 0;
        verbose = 0;
        do_nmf = 0;
        enable_cuda = false;
        nBlocks = 32;
        nThreadsPerBlock = 256;
    }
};


void ccdr1(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ccdr1_original_float(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ALS(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ALS_multicore(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void calculate_rmse();
void calculate_rmse_directly(mat_t& W, mat_t& H, testset_t& T, int iter, int rank, bool ifALS);
void read_input(const parameter& param, const char* input_file_name, smat_t& R, mat_t& W, mat_t& H, testset_t& T, bool ifALS);

#endif
