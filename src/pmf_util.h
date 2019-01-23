#ifndef _PMF_UTIL_H_
#define _PMF_UTIL_H_

void ccdr1(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ccdr1_original_float(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ALS(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void ALS_multicore(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);
void calculate_rmse();
void calculate_rmse_directly(mat_t& W, mat_t& H, testset_t& T, int iter, int rank, bool ifALS);
void read_input(const parameter& param, const char* input_file_name, smat_t& R, mat_t& W, mat_t& H, testset_t& T, bool ifALS);

#endif