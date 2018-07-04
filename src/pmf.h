#ifndef _PMF_H_
#define _PMF_H_

#include "util.h"

enum { CCDR1 };
enum { BOLDDRIVER, EXPDECAY };


class parameter {
public:
    int solver_type;
    int k;
    int threads;
    int maxiter, maxinneriter;
    float lambda;
    float rho;
    float eps;   // for the fundec stop-cond in ccdr1
    float eta0, betaup, betadown;  // learning rate parameters used in DSGD
    int lrate_method, num_blocks;
    int do_predict, verbose;
    int do_nmf;  // non-negative matrix factorization
    bool enable_cuda;
    int nBlocks;
    int nThreadsPerBlock;

    parameter() {
        solver_type = CCDR1;
        k = 10;
        rho = 1e-3f;
        maxiter = 5;
        maxinneriter = 5;
        lambda = 0.1f;
        threads = 4;
        eps = 1e-3f;
        eta0 = 1e-3f;     // initial eta0
        betaup = 1.05f;
        betadown = 0.5f;
        num_blocks = 30;  // number of blocks used in dsgd
        lrate_method = BOLDDRIVER;
        do_predict = 0;
        verbose = 0;
        do_nmf = 0;
        enable_cuda = false;
        nBlocks = 16;
        nThreadsPerBlock = 32;
    }
};


void ccdr1(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ccdr1_original_float(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ALS(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ALS_multicore(smat_t& R, mat_t& W, mat_t& H, parameter& param);

void calculate_rmse();
void calculate_rmse_directly(float** W, float** H, int iter, int rank);
void read_input(const parameter& param, const char* input_file_name, smat_t& R, mat_t& W, mat_t& H, testset_t& T, bool ifALS);

#endif
