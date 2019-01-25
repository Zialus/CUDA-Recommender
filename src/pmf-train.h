#ifndef _PMF_TRAIN_H_
#define _PMF_TRAIN_H_

void generate_file_pointers(const parameter& param, char* test_file_name, char* train_file_name, char* model_file_name,
                            char* output_file_name);

void ccdr1(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);

void ccdr1_original_float(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);

void ALS(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);

void ALS_multicore(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);

void calculate_rmse(FILE* model_fp, FILE* test_fp, FILE* output_fp);

void calculate_rmse_directly(mat_t& W, mat_t& H, testset_t& T, int iter, int rank, bool ifALS);

void read_input(const parameter& param, smat_t& R, testset_t& T, bool ifALS);

#endif
