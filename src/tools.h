#ifndef TOOLS_H
#define TOOLS_H

#include "util.h"

void load(const char* srcdir, smat_t& R, testset_t& T, bool ifALS);
void save_mat_t(mat_t A, FILE* fp, bool row_major = true);
mat_t load_mat_t(FILE* fp, bool row_major = true);
void initial(mat_t& X, long n, long k);
void initial_col(mat_t& X, long k, long n);
float dot(const vec_t& a, const vec_t& b);
double dot(const mat_t& W, int i, const mat_t& H, int j, bool ifALS);
float dot(const mat_t& W, int i, const vec_t& H_j);
float norm(const vec_t& a);
float norm(const mat_t& M);
float calloss(const smat_t& R, const mat_t& W, const mat_t& H);
float calobj(const smat_t& R, const mat_t& W, const mat_t& H, float lambda, bool iscol = false);
double calrmse(testset_t& testset, const mat_t& W, const mat_t& H, bool ifALS, bool iscol = false);
double calrmse_r1(testset_t& testset, vec_t& Wt, vec_t& H_t);
double calrmse_r1(testset_t& testset, vec_t& Wt, vec_t& Ht, vec_t& oldWt, vec_t& oldHt);

#endif //TOOLS_H
