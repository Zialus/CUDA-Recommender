#ifndef EXTRAS_H
#define EXTRAS_H

#include "pmf.h"
#include "util.h"
#include "tools.h"

void generate_file_pointers(const parameter& param, char* test_file_name, char* train_file_name, char* model_file_name,
                            char* output_file_name);
void exit_with_help();

parameter parse_command_line(int argc, char** argv);

void calculate_rmse(FILE* model_fp, FILE* test_fp, FILE* output_fp);

void calculate_rmse_directly(mat_t& W, mat_t& H, testset_t& T, int iter, int rank, bool ifALS);

void read_input(const parameter& param, smat_t& R, testset_t& T, bool ifALS);

#endif //EXTRAS_H
