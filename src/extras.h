#ifndef EXTRAS_H
#define EXTRAS_H

#include "pmf.h"
#include "util.h"
#include "tools.h"

void open_files(const char* test_file_name, const char* model_file_name, const char* output_file_name, FILE*& test_fp,
                FILE*& output_fp, FILE*& model_fp);

void generate_file_pointers(const parameter& param, char* test_file_name, char* train_file_name, char* model_file_name,
                            char* output_file_name);

void exit_with_help();

parameter parse_command_line(int argc, char** argv);

void calculate_rmse_from_file(FILE* model_fp, FILE* test_fp, FILE* output_fp);

void calculate_rmse_directly(MatData& W, MatData& H, TestData& T, int rank, bool ifALS);

void golden_compare(MatData W, MatData W_ref, unsigned k, unsigned m);

void print_matrix(MatData M, unsigned k, unsigned n);

void show_final_matrix(MatData& W, MatData& H, int rank, unsigned n, unsigned m, bool ifALS);

#endif //EXTRAS_H
