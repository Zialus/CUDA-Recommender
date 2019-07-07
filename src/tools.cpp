#include "tools.h"

void load(const char* srcdir, SparseMatrix& R, TestData& T) {
    char filename[1024];
    snprintf(filename, sizeof(filename), "%s/meta_modified_all", srcdir);
    FILE* fp = fopen(filename, "r");

    if (fp == nullptr) {
        printf("Can't open meta input file.\n");
        exit(EXIT_FAILURE);
    }

    char buf[1024];

    long m;
    long n;
    long nnz;
    CHECK_FSCAN(fscanf(fp, "%ld %ld %ld", &m, &n, &nnz), 3);

    char binary_filename_val[1024];
    char binary_filename_row[1024];
    char binary_filename_col[1024];
    char binary_filename_rowptr[1024];
    char binary_filename_colidx[1024];
    char binary_filename_csrval[1024];
    char binary_filename_colptr[1024];
    char binary_filename_rowidx[1024];
    char binary_filename_cscval[1024];

    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_val, sizeof(binary_filename_val), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_row, sizeof(binary_filename_row), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_col, sizeof(binary_filename_col), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_rowptr, sizeof(binary_filename_rowptr), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_colidx, sizeof(binary_filename_colidx), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_csrval, sizeof(binary_filename_csrval), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_colptr, sizeof(binary_filename_colptr), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_rowidx, sizeof(binary_filename_rowidx), "%s/%s", srcdir, buf));
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    CHECK_SNPRINTF(snprintf(binary_filename_cscval, sizeof(binary_filename_cscval), "%s/%s", srcdir, buf));

    auto t0 = std::chrono::high_resolution_clock::now();
    R.initialize_matrix(m, n, nnz);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t1 - t0;
    std::cout << "[info] Alloc TIMER: " << deltaT.count() << "s.\n";


    auto t2 = std::chrono::high_resolution_clock::now();

    R.read_binary_file(binary_filename_rowptr, binary_filename_colidx, binary_filename_csrval,
                       binary_filename_colptr, binary_filename_rowidx, binary_filename_cscval);
    auto t3 = std::chrono::high_resolution_clock::now();
    deltaT = t3 - t2;
    std::cout << "[info] Train TIMER: " << deltaT.count() << "s.\n";

    unsigned long nnz_test;
    CHECK_FSCAN(fscanf(fp, "%lu", &nnz_test),1);

    char binary_filename_val_test[2048];
    char binary_filename_row_test[2048];
    char binary_filename_col_test[2048];

    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_val_test, sizeof(binary_filename_val_test), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_row_test, sizeof(binary_filename_row_test), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_col_test, sizeof(binary_filename_col_test), "%s/%s", srcdir, buf);

    auto t4 = std::chrono::high_resolution_clock::now();
    T.read_binary_file(m, n, nnz_test, binary_filename_val_test, binary_filename_row_test, binary_filename_col_test);
    auto t5 = std::chrono::high_resolution_clock::now();
    deltaT = t5 - t4;
    std::cout << "[info] Tests TIMER: " << deltaT.count() << "s.\n";

    fclose(fp);
}

// Save a mat_t A to a file.
// row_major = true: A is stored in row_major order,
// row_major = false: A is stored in col_major order.
void save_mat_t(MatData A, FILE* fp, bool row_major) {
    if (fp == nullptr) {
        fprintf(stderr, "output stream is not valid.\n");
        exit(EXIT_FAILURE);
    }
    size_t m = row_major ? A.size() : A[0].size();
    size_t n = row_major ? A[0].size() : A.size();

    fwrite(&m, sizeof(long), 1, fp);
    fwrite(&n, sizeof(long), 1, fp);
    //printf("passed\n");
    VecData buf(m * n);
    //printf("passed-buffer\n");
    if (row_major) {
        size_t idx = 0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                buf[idx++] = A[i][j];
            }
        }
    } else {
        size_t idx = 0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                buf[idx++] = A[j][i];
            }
        }
    }
    fwrite(&buf[0], sizeof(float), m * n, fp);
}

// Load a matrix from a file and return a mat_t matrix.
// row_major = true: the returned A is stored in row_major order,
// row_major = false: the returned A is stored in col_major order.
MatData load_mat_t(FILE* fp, bool row_major) {
    if (fp == nullptr) {
        fprintf(stderr, "input stream is not valid.\n");
        exit(EXIT_FAILURE);
    }
    unsigned long m, n;
    CHECK_FREAD(fread(&m, sizeof(unsigned long), 1, fp), 1);
    CHECK_FREAD(fread(&n, sizeof(unsigned long), 1, fp), 1);
    VecData buf(m * n);
    CHECK_FREAD(fread(&buf[0], sizeof(float), m * n, fp), 1);
    MatData A;
    if (row_major) {
        A = MatData(m, VecData(n));
        size_t idx = 0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A[i][j] = buf[idx++];
            }
        }
    } else {
        A = MatData(n, VecData(m));
        size_t idx = 0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A[j][i] = buf[idx++];
            }
        }
    }
    return A;
}

void initial(MatData& X, long n, long k) {
    X = MatData(n, VecData(k));
    srand(0L);
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < k; ++j) {
            X[i][j] = 0.1f * (float(rand()) / RAND_MAX);
        }
    }
}

void initial_col(MatData& X, long k, long n) {
    X = MatData(k, VecData(n));
    srand(0L);
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < k; ++j) {
            X[j][i] = 0.1f * (float(rand()) / RAND_MAX) + 0.001f;
        }
    }
}

float dot(const VecData& a, const VecData& b) {
    float ret = 0;
#pragma omp parallel for
    for (long i = a.size() - 1; i >= 0; --i) {
        ret += a[i] * b[i];
    }
    return ret;
}

double dot(const MatData& W, const long i, const MatData& H, const long j, bool ifALS) {
    double ret = 0;
    if (ifALS) {
        long k = W[0].size();
        for (int t = 0; t < k; ++t) {
            ret += W[i][t] * H[j][t];
        }
    } else {
        long k = W.size();
        for (int t = 0; t < k; ++t) {
            ret += W[t][i] * H[t][j];
        }
    }
    return ret;
}

float dot(const MatData& W, const int i, const VecData& H_j) {
    long k = H_j.size();
    float ret = 0;
    for (int t = 0; t < k; ++t) {
        ret += W[t][i] * H_j[t];
    }
    return ret;
}

float norm(const VecData& a) {
    float ret = 0;
    for (long i = a.size() - 1; i >= 0; --i) {
        ret += a[i] * a[i];
    }
    return ret;
}

float norm(const MatData& M) {
    float reg = 0;
    for (long i = M.size() - 1; i >= 0; --i) { reg += norm(M[i]); }
    return reg;
}

float calloss(const SparseMatrix& R, const MatData& W, const MatData& H) {
    float loss = 0;
    for (long c = 0; c < R.cols; ++c) {
        for (long idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
            float diff = -R.get_csc_val()[idx];
            diff += dot(W[R.get_csc_row_indx()[idx]], H[c]);
            loss += diff * diff;
        }
    }
    return loss;
}

double calrmse(TestData& T, const MatData& W, const MatData& H, bool ifALS, bool iscol) {
    long nnz = T.nnz;
    double rmse = 0;
    for (long idx = 0; idx < nnz; ++idx) {
        double err = -T.getTestVal()[idx];
        if (iscol) {
            err += dot(W, T.getTestRow()[idx], H, T.getTestCol()[idx], ifALS);
        } else {
            err += dot(W[T.getTestRow()[idx]], H[T.getTestCol()[idx]]);
        }
        rmse += err * err;
    }
    return sqrt(rmse / nnz);
}

double calrmse_r1(TestData& T, VecData& Wt, VecData& Ht) {
    long nnz = T.nnz;
    double rmse = 0;
#pragma omp parallel for reduction(+:rmse)
    for (int idx = 0; idx < nnz; ++idx) {
        T.getTestVal()[idx] -= Wt[T.getTestRow()[idx]] * Ht[T.getTestCol()[idx]];
        rmse += T.getTestVal()[idx] * T.getTestVal()[idx];
    }
    return sqrt(rmse / nnz);
}

double calrmse_r1(TestData& T, VecData& Wt, VecData& Ht, VecData& oldWt, VecData& oldHt) {
    long nnz = T.nnz;
    double rmse = 0;
#pragma omp parallel for reduction(+:rmse)
    for (int idx = 0; idx < nnz; ++idx) {
        T.getTestVal()[idx] -= Wt[T.getTestRow()[idx]] * Ht[T.getTestCol()[idx]] - oldWt[T.getTestRow()[idx]] * oldHt[T.getTestCol()[idx]];
        rmse += T.getTestVal()[idx] * T.getTestVal()[idx];
    }
    return sqrt(rmse / nnz);
}
