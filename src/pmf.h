#ifndef PMF_H
#define PMF_H

#include "pmf_util.h"

enum class solvertype {CCD, ALS};

class parameter {
public:
    solvertype solver_type;
    int k;
    int threads;
    int maxiter;
    int maxinneriter;
    float lambda;
    float eps;
    int do_predict;
    int verbose;
    int do_nmf;
    bool enable_cuda;
    bool enable_omp;
    unsigned nBlocks;
    unsigned nThreadsPerBlock;
    char src_dir[1024];

    parameter() {
        solver_type = solvertype::CCD;
        k = 10;
        threads = 4;
        maxiter = 5;
        maxinneriter = 5;
        lambda = 0.1f;
        eps = 1e-3f;  // for the fundec stop-cond in ccdr1
        do_predict = 0;
        verbose = 0;
        do_nmf = 0;  // non-negative matrix factorization
        enable_cuda = false;
        enable_omp = false;
        nBlocks = 32;
        nThreadsPerBlock = 256;
        snprintf(src_dir, sizeof(src_dir), "../data/simple");
    }
};

#endif //PMF_H
