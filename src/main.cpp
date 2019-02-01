#include "extras.h"
#include "ALS_CUDA.h"
#include "CCD_CUDA.h"
#include "ALS.h"
#include "CCD.h"

int main(int argc, char* argv[]) {
    auto t7 = std::chrono::high_resolution_clock::now();

    parameter param = parse_command_line(argc, argv);

    char test_file_name[2048];
    char train_file_name[2048];
    char model_file_name[2048];
    char output_file_name[2048];

    generate_file_pointers(param, test_file_name, train_file_name, model_file_name, output_file_name);
    printf("input: %s | model: %s | test: %s | output: %s\n",
           train_file_name, model_file_name, test_file_name, output_file_name);

    FILE* test_fp;
    FILE* output_fp;
    FILE* model_fp;

    open_files(test_file_name, model_file_name, output_file_name, test_fp, output_fp, model_fp);

    smat_t R;
    testset_t T;

    mat_t W;
    mat_t H;

    mat_t W_ref;
    mat_t H_ref;

    bool ifALS;

    switch (param.solver_type) {
        case solvertype::CCD: {
            read_input(param, R, T, false);

            initial_col(W, param.k, R.rows);
            initial_col(H, param.k, R.cols);

            initial_col(W_ref, param.k, R.rows);
            initial_col(H_ref, param.k, R.cols);

            printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));

            puts("----------=CCD CUDA START=------");
            if (param.enable_cuda) {
                double time1 = omp_get_wtime();
                kernel_wrapper_ccdpp_NV(R, T, W, H, param);
                double time2 = omp_get_wtime();
                printf("CCD CUDA run time: %lf secs\n", time2 - time1);
            }
            puts("----------=CCD non-CUDA START=------");
            double time1 = omp_get_wtime();
            ccdr1_OMP(R, W_ref, H_ref, T, param);
            double time2 = omp_get_wtime();
            printf("CCD non-CUDA run time: %lf secs\n", time2 - time1);

            ifALS = false;
            break;
        }
        case solvertype::ALS: {
            read_input(param, R, T, true);

            initial_col(W, R.rows, param.k);
            initial_col(H, R.cols, param.k);

            initial_col(W_ref, R.rows, param.k);
            initial_col(H_ref, R.cols, param.k);

            printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));

            puts("----------=ALS CUDA START=------");
            if (param.enable_cuda) {
                double time1 = omp_get_wtime();
                printf("CUDA enabled version.\n");
                kernel_wrapper_als_NV(R, T, W, H, param);
                double time2 = omp_get_wtime();
                printf("ALS CUDA run time: %lf secs\n", time2 - time1);
            }
            puts("----------=ALS non-CUDA START=------");
            double time1 = omp_get_wtime();
            ALS_OMP(R, W_ref, H_ref, T, param);
            double time2 = omp_get_wtime();
            printf("ALS non-CUDA run time: %lf secs\n", time2 - time1);

            ifALS = true;
            break;
        }
        default: {
            fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
            exit(EXIT_FAILURE);
        }
    }

//    calculate_rmse_directly(W, H, T, 5, param.k, ifALS);

    save_mat_t(W, model_fp, ifALS);
    save_mat_t(H, model_fp, ifALS);

    calculate_rmse(model_fp, test_fp, output_fp);

    fclose(model_fp);
    fclose(output_fp);
    fclose(test_fp);

    auto t8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT78 = t8 - t7;
    printf("Total Time: %lf!\n", deltaT78.count());

    return EXIT_SUCCESS;
}


