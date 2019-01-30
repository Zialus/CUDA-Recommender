#include "extras.h"

int main(int argc, char* argv[]) {
    auto t7 = std::chrono::high_resolution_clock::now();
    parameter param = parse_command_line(argc, argv);

    char test_file_name[2048], train_file_name[2048], model_file_name[2048], output_file_name[2048];
    generate_file_pointers(param, test_file_name, train_file_name, model_file_name, output_file_name);
    printf("input: %s | model: %s | test: %s | output: %s\n",
           train_file_name, model_file_name, test_file_name, output_file_name);

    FILE* test_fp = fopen(test_file_name, "r");
    if (test_fp == nullptr) {
        fprintf(stderr, "can't open test file %s\n", test_file_name);
        exit(EXIT_FAILURE);
    }
    FILE* output_fp = fopen(output_file_name, "w+b");
    if (output_fp == nullptr) {
        fprintf(stderr, "can't open output file %s\n", output_file_name);
        exit(EXIT_FAILURE);
    }
    FILE* model_fp = fopen(model_file_name, "w+b");
    if (model_fp == nullptr) {
        fprintf(stderr, "can't open model file %s\n", model_file_name);
        exit(EXIT_FAILURE);
    }

    smat_t R;
    mat_t W;
    mat_t H;
    testset_t T;

    switch (param.solver_type) {
        case solvertype::CCD:
            read_input(param, R, T, false);

            initial_col(W, param.k, R.rows);
            initial_col(H, param.k, R.cols);

            fprintf(stdout, "CCD\n");
            printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));

            run_ccdr1(param, R, W, H, T);

            save_mat_t(W, model_fp, false);
            save_mat_t(H, model_fp, false);
            break;
        case solvertype::ALS:
            read_input(param, R, T, true);

            initial_col(W, R.rows, param.k);
            initial_col(H, R.cols, param.k);

            fprintf(stdout, "ALS\n");
            printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));

            run_ALS(param, R, W, H, T);

            save_mat_t(W, model_fp, true);
            save_mat_t(H, model_fp, true);

            break;
        default:
            fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
            break;
    }

    calculate_rmse(model_fp, test_fp, output_fp);

    fclose(model_fp);
    fclose(output_fp);
    fclose(test_fp);

    auto t8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT78 = t8 - t7;

    printf("Total Time: %lf!\n", deltaT78.count());

    return EXIT_SUCCESS;
}
