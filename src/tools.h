#ifndef TOOLS_H
#define TOOLS_H

#include "pmf_util.h"

void load(const char* srcdir, SparseMatrix& R, TestData& T);
void save_mat_t(MatData A, FILE* fp, bool row_major = true);
MatData load_mat_t(FILE* fp, bool row_major = true);
void initial(MatData& X, long n, long k);
void initial_col(MatData& X, long k, long n);
float dot(const VecData& a, const VecData& b);
double dot(const MatData& W, long i, const MatData& H, long j, bool ifALS);
float dot(const MatData& W, int i, const VecData& H_j);
float norm(const VecData& a);
float norm(const MatData& M);
float calloss(const SparseMatrix& R, const MatData& W, const MatData& H);
double calrmse(TestData& T, const MatData& W, const MatData& H, bool ifALS, bool iscol = false);
double calrmse_r1(TestData& T, VecData& Wt, VecData& H_t);
double calrmse_r1(TestData& T, VecData& Wt, VecData& Ht, VecData& oldWt, VecData& oldHt);

#endif //TOOLS_H
