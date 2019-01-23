#include "pmf.h"
#include "pmf_util.h"

bool with_weights;

FILE* test_fp = nullptr;
FILE* model_fp = nullptr;
FILE* output_fp = nullptr;

void exit_with_help()
{
    printf(
            "Usage: omp-pmf-train [options] data_dir [model_filename]\n"
            "options:\n"
            "    -s type : set type of solver (default 0)\n"
            "        0 -- CCDR1 with fundec stopping condition\n"
            "    -k rank : set the rank (default 10)\n"
            "    -n threads : set the number of threads (default 4)\n"
            "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
            "    -t max_iter: set the number of iterations (default 5)\n"
            "    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"
            "    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"
            "    -p do_predict: do prediction or not (default 0)\n"
            "    -q verbose: show information or not (default 0)\n"
            "    -N do_nmf: do nmf (default 0)\n"
            "    -runOriginal: Flag to run libpmf original implementation\n"
            "    -Cuda: Flag to enable cuda\n"
            "    -nBlocks: Number of blocks on cuda (default 16)\n"
            "    -nThreadsPerBlock: Number of threads per block on cuda (default 32)\n"
            "    -ALS: Flag to enable ALS algorithm, if not present CCD++ is used\n"
    );

    exit(1);
}

parameter parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *test_file_name, char* output_file_name)
{
    parameter param;   // default values have been set by the constructor
    with_weights = false;
    int i;

    // parse options
    for(i=1;i<argc;i++)
    {
        if (argv[i][0] != '-'){
            break;
        }
        if (++i >= argc){
            exit_with_help();
        }
        if (strcmp(argv[i - 1], "-nBlocks") == 0){
            param.nBlocks = atoi(argv[i]);
        }
        else if (strcmp(argv[i - 1], "-nThreadsPerBlock") == 0){
            param.nThreadsPerBlock = atoi(argv[i]);
        }
        else if (strcmp(argv[i - 1], "-Cuda") == 0){
            param.enable_cuda = true;
            --i;
        }
        else if (strcmp(argv[i - 1], "-CCD") == 0){
            param.solver_type = solvertype::CCD;
            --i;
        }else if (strcmp(argv[i - 1], "-ALS") == 0){
            param.solver_type = solvertype::ALS;
            --i;
        }
        else{
            switch (argv[i - 1][1])
            {

                case 'k':
                    param.k = atoi(argv[i]);
                    break;

                case 'n':
                    param.threads = atoi(argv[i]);
                    break;

                case 'l':
                    param.lambda = (float) atof(argv[i]);
                    break;

                case 't':
                    param.maxiter = atoi(argv[i]);
                    break;

                case 'T':
                    param.maxinneriter = atoi(argv[i]);
                    break;

                case 'e':
                    param.eps = (float) atof(argv[i]);
                    break;

                case 'p':
                    param.do_predict = atoi(argv[i]);
                    break;

                case 'q':
                    param.verbose = atoi(argv[i]);
                    break;

                case 'N':
                    param.do_nmf = (atoi(argv[i]) == 1);
                    break;

                default:
                    fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                    exit_with_help();
                    break;
            }
        }

    }

    if (param.do_predict!=0)
        param.verbose = 1;

    if(i>=argc)
        exit_with_help();

    sprintf(input_file_name, "%s", argv[i]);

    sprintf(model_file_name, "%s", argv[i+1]);

    sprintf(test_file_name, "%s", argv[i+2]);

    sprintf(output_file_name, "%s", argv[i+3]);

    return param;
}


void run_ccdr1(parameter& param, const char* input_file_name) {
    smat_t R;
    mat_t W;
    mat_t H;
    testset_t T;

    read_input(param, input_file_name, R, W, H, T, false);

//    printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));

    puts("----------=CCD START=------");
    double time = omp_get_wtime();
    ccdr1(R, W, H, T, param);
    printf("Wall-time: %lf secs\n", omp_get_wtime() - time);
    puts("----------=CCD END=--------");

    if (model_fp) {
        save_mat_t(W, model_fp, false);
        save_mat_t(H, model_fp, false);
    }

}

void run_ALS(parameter& param, const char* input_file_name) {
    smat_t R;
    mat_t W;
    mat_t H;
    testset_t T;

    read_input(param, input_file_name, R, W, H, T, true);

//    printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));

    puts("----------=ALS START=------");
    double time = omp_get_wtime();
    ALS(R, W, H, T, param);
    printf("Wall-time: %lg secs\n", omp_get_wtime() - time);
    puts("----------=ALS END=--------");

    if (model_fp) {
        save_mat_t(W, model_fp, true);
        save_mat_t(H, model_fp, true);
    }
}

void read_input(const parameter& param, const char* input_file_name, smat_t& R, mat_t& W, mat_t& H, testset_t& T, bool ifALS) {
    puts("----------=INPUT START=------");
    puts("Starting to read inout...");
    double time1 = omp_get_wtime();
    load(input_file_name, R, T, ifALS, with_weights);
    printf("Input loaded in: %lg secs\n", omp_get_wtime() - time1);
    puts("----------=INPUT END=--------");

    if (ifALS) {
        initial_col(W, R.rows, param.k);
        initial_col(H, R.cols, param.k);
    } else { // CDD
        initial_col(W, param.k, R.rows);
        initial_col(H, param.k, R.cols);
    }
}

int main(int argc, char* argv[]) {

    char input_file_name[1024];
    char model_file_name[1024];
    char test_file_name[1024];
    char output_file_name[1024];

    parameter param = parse_command_line(argc, argv, input_file_name, model_file_name, test_file_name, output_file_name);

//     printf("-----------\n");
//     printf("input: %s\n",input_file_name);
//     printf("model: %s\n",model_file_name);
//     printf("test: %s\n",test_file_name);
//     printf("output: %s\n",output_file_name);
//     printf("-----------\n");


    test_fp = fopen(test_file_name, "r");
    if (test_fp == nullptr) {
        fprintf(stderr, "can't open test file %s\n", test_file_name);
        exit(EXIT_FAILURE);
    }

    output_fp = fopen(output_file_name, "w+b");
    if (output_fp == nullptr) {
        fprintf(stderr, "can't open output file %s\n", output_file_name);
        exit(EXIT_FAILURE);
    }

    model_fp = fopen(model_file_name, "w+b");
    if (model_fp == nullptr) {
        fprintf(stderr, "can't open model file %s\n", model_file_name);
        exit(EXIT_FAILURE);
    }

    switch (param.solver_type) {
        case solvertype::CCD:
            fprintf(stdout, "CCD\n");
            run_ccdr1(param, input_file_name);
            break;
        case solvertype::ALS:
            fprintf(stdout, "ALS\n");
            run_ALS(param, input_file_name);
            break;
        default:
            fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
            break;
    }

    puts("Final RMSE Calculation");
    calculate_rmse();

    fclose(model_fp);
    fclose(output_fp);
    fclose(test_fp);
    return EXIT_SUCCESS;
}

void calculate_rmse() {

    rewind(model_fp);
    rewind(test_fp);
    rewind(output_fp);

    double time = omp_get_wtime();

    mat_t W = load_mat_t(model_fp, true);
    mat_t H = load_mat_t(model_fp, true);

    unsigned long rank = W[0].size();
    if (rank == 0) {
        fprintf(stderr, "Matrix is emty!\n");
        exit(1);
    }
    int i;
    int j;
    double v;
    double rmse = 0;
    size_t num_insts = 0;

    while (fscanf(test_fp, "%d %d %lf", &i, &j, &v) != EOF) {
        double pred_v = 0;
//#pragma omp parallel for  reduction(+:pred_v)
        for (unsigned long t = 0; t < rank; t++) {
            pred_v += W[i - 1][t] * H[j - 1][t];
        }
        num_insts++;
        rmse += (pred_v - v) * (pred_v - v);
        fprintf(output_fp, "%lf\n", pred_v);
    }

    rmse = sqrt(rmse / num_insts);
    printf("Test RMSE = %f , calculated in %lfs\n", rmse, omp_get_wtime() - time);
}

void calculate_rmse_directly(mat_t& W, mat_t& H, testset_t& T, int iter, int rank, bool ifALS) {

    rewind(test_fp);

    double time = omp_get_wtime();

    int i;
    int j;
    double v;
    double rmse = 0;
    size_t num_insts = 0;

    while (fscanf(test_fp, "%d %d %lf", &i, &j, &v) != EOF) {
        double pred_v = 0;

        if (ifALS) {
//#pragma omp parallel for  reduction(+:pred_v)
            for (int t = 0; t < rank; t++) {
                pred_v += W[i - 1][t] * H[j - 1][t];
            }
        } else {
//#pragma omp parallel for  reduction(+:pred_v)
            for (int t = 0; t < rank; t++) {
                pred_v += W[t][i - 1] * H[t][j - 1];
            }
        }

        num_insts++;
        rmse += (pred_v - v) * (pred_v - v);
    }

    rmse = sqrt(rmse / num_insts);
    printf("Test RMSE = %f , for iter %d, calculated in %lfs\n", rmse, iter, omp_get_wtime() - time);
}
