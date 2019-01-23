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

#endif
