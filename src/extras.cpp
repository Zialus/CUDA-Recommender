#include "extras.h"

void open_files(const char* test_file_name, const char* model_file_name, const char* output_file_name, FILE*& test_fp,
                FILE*& output_fp, FILE*& model_fp) {
    test_fp = fopen(test_file_name, "r");
    output_fp = fopen(output_file_name, "w+b");
    model_fp = fopen(model_file_name, "w+b");

    if (test_fp == nullptr) {
        fprintf(stderr, "can't open test file %s\n", test_file_name);
        exit(EXIT_FAILURE);
    }

    if (output_fp == nullptr) {
        fprintf(stderr, "can't open output file %s\n", output_file_name);
        exit(EXIT_FAILURE);
    }

    if (model_fp == nullptr) {
        fprintf(stderr, "can't open model file %s\n", model_file_name);
        exit(EXIT_FAILURE);
    }
}

void generate_file_pointers(const parameter& param, char* test_file_name, char* train_file_name, char* model_file_name,
                            char* output_file_name) {
    char meta_filename[1024];
    sprintf(meta_filename, "%s/meta", param.src_dir);
    FILE* fp = fopen(meta_filename, "r");
    if (fp == nullptr) {
        fprintf(stderr, "Can't open meta input file %s\n", meta_filename);
        exit(EXIT_FAILURE);
    }

    char buf_train[1024], buf_test[1024];
    unsigned m, n, nnz, nnz_test;
    fscanf(fp, "%u %u", &m, &n);
    fscanf(fp, "%u %1023s", &nnz, buf_train);
    fscanf(fp, "%u %1023s", &nnz_test, buf_test);
    sprintf(test_file_name, "%s/%s", param.src_dir, buf_test);
    sprintf(train_file_name, "%s/%s", param.src_dir, buf_train);
    sprintf(model_file_name, "%s/%s", param.src_dir, "model");
    sprintf(output_file_name, "%s/%s", param.src_dir, "output");
    fclose(fp);
}

void exit_with_help() {
    printf(
            "Usage: omp-pmf-train [options] data_dir [model_filename]\n"
            "options:\n"
            "    -k rank : set the rank (default 10)\n"
            "    -n threads : set the number of threads (default 4)\n"
            "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
            "    -t max_iter: set the number of iterations (default 5)\n"
            "    -T max_inner_iter: set the number of inner iterations used in CCDR1 (default 5)\n"
            "    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"
            "    -p do_predict: do prediction or not (default 0)\n"
            "    -q verbose: show information or not (default 0)\n"
            "    -N do_nmf: do nmf (default 0)\n"
            "    -CUDA: Flag to enable CUDA\n"
            "    -nBlocks: Number of blocks on CUDA (default 32)\n"
            "    -nThreadsPerBlock: Number of threads per block on CUDA (default 256)\n"
            "    -ALS: Flag to enable ALS algorithm, if not present CCD++ is used\n"
    );

    exit(EXIT_FAILURE);
}

parameter parse_command_line(int argc, char** argv) {
    parameter param{};

    int i;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            break;
        }
        if (++i >= argc) {
            exit_with_help();
        }
        if (strcmp(argv[i - 1], "-nBlocks") == 0) {
            param.nBlocks = atoi(argv[i]);
        } else if (strcmp(argv[i - 1], "-nThreadsPerBlock") == 0) {
            param.nThreadsPerBlock = atoi(argv[i]);
        } else if (strcmp(argv[i - 1], "-CUDA") == 0) {
            param.enable_cuda = true;
            --i;
        } else if (strcmp(argv[i - 1], "-ALS") == 0) {
            param.solver_type = solvertype::ALS;
            --i;
        } else {
            switch (argv[i - 1][1]) {
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

    if (param.do_predict != 0) {
        param.verbose = 1;
    }

    if (i >= argc) {
        exit_with_help();
    }

    sprintf(param.src_dir, "%s", argv[i]);

    return param;
}

void calculate_rmse(FILE* model_fp, FILE* test_fp, FILE* output_fp) {

    rewind(model_fp);
    rewind(test_fp);
    rewind(output_fp);

    double time = omp_get_wtime();

    mat_t W = load_mat_t(model_fp, true);
    mat_t H = load_mat_t(model_fp, true);

    unsigned long rank = W[0].size();
    if (rank == 0) {
        fprintf(stderr, "Matrix is empty!\n");
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

    double time = omp_get_wtime();

    double rmse = 0;
    size_t num_insts = 0;

    long nnz = T.nnz;

    for (long idx = 0; idx < nnz; ++idx) {
        long i = T.test_row[idx];
        long j = T.test_col[idx];
        double v = T.test_val[idx];

        double pred_v = 0;
        if (ifALS) {
//#pragma omp parallel for  reduction(+:pred_v)
            for (int t = 0; t < rank; t++) {
                pred_v += W[i][t] * H[j][t];
            }
        } else {
//#pragma omp parallel for  reduction(+:pred_v)
            for (int t = 0; t < rank; t++) {
                pred_v += W[t][i] * H[t][j];
            }
        }

        num_insts++;
        rmse += (pred_v - v) * (pred_v - v);
    }

    rmse = sqrt(rmse / num_insts);
    printf("Test RMSE = %f , for iter %d, calculated in %lfs\n", rmse, iter, omp_get_wtime() - time);
}

void read_input(const parameter& param, smat_t& R, testset_t& T, bool ifALS) {
    puts("----------=INPUT START=------");
    double time1 = omp_get_wtime();
    load(param.src_dir, R, T, ifALS);
    double time2 = omp_get_wtime();
    printf("Input loaded in: %lf secs\n", time2 - time1);
    puts("----------=INPUT END=--------");
}
