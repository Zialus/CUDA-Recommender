#ifndef _PMF_H_ORIGINAL
#define _PMF_H_ORIGINAL

#include "util_original.h"

enum { CCDR1_Double };
enum { BOLDDRIVER_Double, EXPDECAY_Double };

class parameter_Double {
public:
    int solver_type;
    int k;
    int threads;
    int maxiter, maxinneriter;
    double lambda;
    double rho;
    double eps;                        // for the fundec stop-cond in ccdr1
    double eta0, betaup, betadown;  // learning rate parameters used in DSGD
    int lrate_method, num_blocks;
    int do_predict, verbose;
    int do_nmf;  // non-negative matrix factorization
    parameter_Double() {
        solver_type = CCDR1_Double;
        k = 10;
        rho = 1e-3;
        maxiter = 5;
        maxinneriter = 5;
        lambda = 0.1;
        threads = 4;
        eps = 1e-3;
        eta0 = 1e-3; // initial eta0
        betaup = 1.05;
        betadown = 0.5;
        num_blocks = 30;  // number of blocks used in dsgd
        lrate_method = BOLDDRIVER_Double;
        do_predict = 0;
        verbose = 0;
        do_nmf = 0;
    }
};


void ccdr1_Double(smat_t_Double& R, mat_t_Double& W, mat_t_Double& H, testset_t_Double& T, parameter_Double& param);


#endif
