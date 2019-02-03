#include "extras.h"
#include "ALS_CUDA.h"
#include "CCD_CUDA.h"
#include "ALS.h"
#include "CCD.h"

std::chrono::duration<double> deltaT12;
std::chrono::duration<double> deltaTAB;
std::chrono::duration<double> deltaT34;

void runCUDA(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters, bool ALS) {
    if (ALS) {
        kernel_wrapper_als_NV(R, T, W, H, parameters);
    } else {
        kernel_wrapper_ccdpp_NV(R, T, W, H, parameters);
    }
}

void runOMP(smat_t& R, testset_t& T, mat_t& W, mat_t& H, parameter& parameters, bool ALS) {
    if (ALS) {
        ALS_OMP(R, W, H, T, parameters);
    } else {
        ccdr1_OMP(R, W, H, T, parameters);
    }
}

void read_input(const parameter& param, smat_t& R, testset_t& T, bool ifALS) {
    puts("------------------------------------------------------------");
    puts("[info] Loading R matrix...");
    auto t3 = std::chrono::high_resolution_clock::now();
    load(param.src_dir, R, T, ifALS);
    auto t4 = std::chrono::high_resolution_clock::now();
    deltaT34 = t4 - t3;
    printf("[info] Loading rating data time: %lf s.\n", deltaT34.count());
    puts("------------------------------------------------------------");
}

int main(int argc, char* argv[]) {
    auto t7 = std::chrono::high_resolution_clock::now();

    parameter param = parse_command_line(argc, argv);

    char test_file_name[2048];
    char train_file_name[2048];
    char model_file_name[2048];
    char output_file_name[2048];

    generate_file_pointers(param, test_file_name, train_file_name, model_file_name, output_file_name);
//    printf("input: %s | model: %s | test: %s | output: %s\n", train_file_name, model_file_name, test_file_name, output_file_name);

    FILE* test_fp;
    FILE* output_fp;
    FILE* model_fp;

    open_files(test_file_name, model_file_name, output_file_name, test_fp, output_fp, model_fp);

    smat_t R;
    testset_t T;

    mat_t W_cuda;
    mat_t H_cuda;

    mat_t W_ref;
    mat_t H_ref;

    bool ifALS;

    switch (param.solver_type) {
        case solvertype::CCD: {
            ifALS = false;
            puts("Picked Version: CCD!");
            read_input(param, R, T, ifALS);

            initial_col(W_cuda, param.k, R.rows);
            initial_col(H_cuda, param.k, R.cols);

            initial_col(W_ref, param.k, R.rows);
            initial_col(H_ref, param.k, R.cols);

            break;
        }
        case solvertype::ALS: {
            ifALS = true;
            puts("Picked Version: ALS!");
            read_input(param, R, T, ifALS);

            initial_col(W_cuda, R.rows, param.k);
            initial_col(H_cuda, R.cols, param.k);

            initial_col(W_ref, R.rows, param.k);
            initial_col(H_ref, R.cols, param.k);

            break;
        }
        default: {
            fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
            exit(EXIT_FAILURE);
        }
    }

    printf("global mean %g\n", R.get_global_mean());

    std::chrono::duration<double> deltaT56{};
    std::chrono::duration<double> deltaT9_10{};
    std::chrono::duration<double> deltaT11_12{};
    std::chrono::duration<double> deltaT13_14{};

    if (param.enable_cuda) {
        puts("------------------------------------------------------------");
        puts("[INFO] Computing with CUDA...");
        auto t5 = std::chrono::high_resolution_clock::now();
        runCUDA(R, T, W_cuda, H_cuda, param, ifALS);
        auto t6 = std::chrono::high_resolution_clock::now();
        deltaT56 = t6 - t5;
        printf("[info] CUDA Training time: %lf s.\n", deltaT56.count());
        puts("------------------------------------------------------------");
        calculate_rmse_directly(W_cuda, H_cuda, T, param.k, ifALS);
    }
    if (param.enable_omp) {
        puts("------------------------------------------------------------");
        puts("[INFO] Computing with OMP...");
        auto t9 = std::chrono::high_resolution_clock::now();
        runOMP(R, T, W_ref, H_ref, param, ifALS);
        auto t10 = std::chrono::high_resolution_clock::now();
        deltaT9_10 = t10 - t9;
        printf("[info] OMP Training time: %lf s.\n", deltaT9_10.count());
        puts("------------------------------------------------------------");
        calculate_rmse_directly(W_ref, H_ref, T, param.k, ifALS);
        puts("------------------------------------------------------------");
    }

    std::cout << "[info] validate the results." << std::endl;
    auto t11 = std::chrono::high_resolution_clock::now();
    if (ifALS) {
        golden_compare(W_cuda, W_ref, R.rows, param.k);
        golden_compare(H_cuda, H_ref, R.cols, param.k);
    } else {
        golden_compare(W_cuda, W_ref, param.k, R.rows);
        golden_compare(H_cuda, H_ref, param.k, R.cols);
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    deltaT11_12 = t12 - t11;
    std::cout << "[info] Validate Time: " << deltaT11_12.count() << " s.\n";

//    save_mat_t(W_cuda, model_fp, ifALS);
//    save_mat_t(H_cuda, model_fp, ifALS);
//
//    calculate_rmse_from_file(model_fp, test_fp, output_fp);

    fclose(model_fp);
    fclose(output_fp);
    fclose(test_fp);

    puts("------------------------------------------------------------");
    auto t8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT78 = t8 - t7;
    std::cout << "Total Time: " << deltaT78.count() << " Parcial Sums:"
              << deltaT12.count() + deltaT34.count() + deltaT56.count() + deltaTAB.count() + deltaT9_10.count()
                 + deltaT11_12.count() + deltaT13_14.count() << " s.\n";

    return EXIT_SUCCESS;
}
