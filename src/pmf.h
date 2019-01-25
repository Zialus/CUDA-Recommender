#ifndef _PMF_H_
#define _PMF_H_

#include "util.h"

enum class solvertype {CCD, ALS};

class parameter {
public:
    solvertype solver_type = solvertype::CCD;
    int k = 10;
    int threads = 4;
    int maxiter = 5;
    int maxinneriter = 5;
    float lambda = 0.1f;
    float eps = 1e-3f;  // for the fundec stop-cond in ccdr1
    int do_predict = 0;
    int verbose = 0;
    int do_nmf = 0;  // non-negative matrix factorization
    bool enable_cuda = false;
    int nBlocks = 32;
    int nThreadsPerBlock = 256;
    char src_dir[1024] = "../data/simple";
};

#endif
