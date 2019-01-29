#ifndef _UTIL_H
#define _UTIL_H

#include <iostream>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <omp.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define SIZEBITS(type, size) sizeof(type)*(size)


class smat_t;

class testset_t;

typedef std::vector<float> vec_t;
typedef std::vector<vec_t> mat_t;

// Comparator for sorting rates into row/column compression storage
class SparseComp {
public:
    const long* row_idx;
    const long* col_idx;

    SparseComp(const long* row_idx_, const long* col_idx_, bool isRCS_ = true) {
        row_idx = (isRCS_) ? row_idx_ : col_idx_;
        col_idx = (isRCS_) ? col_idx_ : row_idx_;
    }

    bool operator()(size_t x, size_t y) const {
        return (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x] <= col_idx[y]));
    }
};

// Sparse matrix format CSC & CSR
// Access column format only when you use it..
class smat_t {
public:
    long rows;
    long cols;
    long nnz;
    long max_row_nnz;
    long max_col_nnz;
    float* val;
    float* val_t;
    size_t nbits_val;
    size_t nbits_val_t;
    long* col_ptr;
    long* row_ptr;
    size_t nbits_col_ptr;
    size_t nbits_row_ptr;
    long* col_nnz;
    long* row_nnz;
    size_t nbits_col_nnz;
    size_t nbits_row_nnz;
    long* row_idx;
    long* col_idx;
    size_t nbits_row_idx;
    size_t nbits_col_idx;
    long* colMajored_sparse_idx;
    size_t nbits_colMajored_sparse_idx;
    bool mem_alloc_by_me;

    smat_t() : mem_alloc_by_me(false) {}

    void print_mat(int host) {
        for (int c = 0; c < cols; ++c) {
            if (col_ptr[c + 1] > col_ptr[c]) {
                printf("%d: %ld at host %d\n", c, col_ptr[c + 1] - col_ptr[c], host);
            }
        }
    }

    void load(long _rows, long _cols, unsigned long _nnz, const char* filename, bool ifALS) {
        rows = _rows, cols = _cols, nnz = _nnz;
        mem_alloc_by_me = true;
        val = MALLOC(float, nnz);
        val_t = MALLOC(float, nnz);
        nbits_val = SIZEBITS(float, nnz);
        nbits_val_t = SIZEBITS(float, nnz);
        row_idx = MALLOC(long, nnz);
        col_idx = MALLOC(long, nnz);  // switch to this for memory
        nbits_row_idx = SIZEBITS(long, nnz);
        nbits_col_idx = SIZEBITS(long, nnz);
        row_ptr = MALLOC(long, rows + 1);
        col_ptr = MALLOC(long, cols + 1);
        nbits_row_ptr = SIZEBITS(long, rows + 1);
        nbits_col_ptr = SIZEBITS(long, cols + 1);
        memset(row_ptr, 0, sizeof(long) * (rows + 1));
        memset(col_ptr, 0, sizeof(long) * (cols + 1));
        if (ifALS) {
            colMajored_sparse_idx = MALLOC(long, nnz);
            nbits_colMajored_sparse_idx = SIZEBITS(long, nnz);
        }

        // a trick here to utilize the space the have been allocated
        std::vector<size_t> perm(_nnz);
        long* tmp_row_idx = col_idx;
        long* tmp_col_idx = row_idx;
        float* tmp_val = val;

        FILE* fp = fopen(filename, "r");
        for (size_t idx = 0; idx < _nnz; idx++) {
            long i;
            long j;
            float v;
            fscanf(fp, "%ld %ld %f", &i, &j, &v);

            row_ptr[i - 1 + 1]++;
            col_ptr[j - 1 + 1]++;
            tmp_row_idx[idx] = i - 1;
            tmp_col_idx[idx] = j - 1;
            tmp_val[idx] = v;
            perm[idx] = idx;
        }
        fclose(fp);

        //for (int i = 0; i < rows + 1; ++i){
        //	printf("R%d  C%d \n", row_ptr[i], col_ptr[i]);
        //}
        //for (int i = 0; i < nnz; ++i){
        //	printf("%d %d %.3f\n", row_idx[i], col_idx[i], val[i]);
        //}
        //printf("\n");
        //for (int i = 0; i < nnz; ++i){
        //	printf("%d %d %.3f\n", tmp_row_idx[i], tmp_col_idx[i], tmp_val[i]);
        //}
        //printf("\n");
        // sort entries into row-majored ordering
        sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));
        // Generate CRS format
        for (size_t idx = 0; idx < _nnz; idx++) {
            val_t[idx] = tmp_val[perm[idx]];
            col_idx[idx] = tmp_col_idx[perm[idx]];
        }

        // Calculate nnz for each row and col
        max_row_nnz = max_col_nnz = 0;
        for (long r = 1; r <= rows; ++r) {
            max_row_nnz = std::max(max_row_nnz, row_ptr[r]);
            row_ptr[r] += row_ptr[r - 1];
        }
        for (long c = 1; c <= cols; ++c) {
            max_col_nnz = std::max(max_col_nnz, col_ptr[c]);
            col_ptr[c] += col_ptr[c - 1];
        }
        // Transpose CRS into CCS matrix
        for (long r = 0; r < rows; ++r) {
            for (long i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
                long c = col_idx[i];
                row_idx[col_ptr[c]] = r;
                val[col_ptr[c]] = val_t[i];
                col_ptr[c]++;
            }
        }
        for (long c = cols; c > 0; --c) { col_ptr[c] = col_ptr[c - 1]; }
        col_ptr[0] = 0;

        if (ifALS) {
            long* mapIDX = MALLOC(long, rows);
            for (long r = 0; r < rows; ++r) {
                mapIDX[r] = row_ptr[r];
            }

            for (long r = 0; r < nnz; ++r) {
                colMajored_sparse_idx[mapIDX[row_idx[r]]] = r;
                ++mapIDX[row_idx[r]];
            }
            //unsigned internalIDX = 0;
            //for (int r = 0; r < rows; ++r){//extremely slow!!
            //	for (int idx = 0; idx < nnz; ++idx){
            //		if (row_idx[idx] == r){
            //			colMajored_sparse_idx[internalIDX] = idx;
            //			++internalIDX;
            //		}
            //	}
            //}

            free(mapIDX);

            //for (int i = 0; i < nnz; ++i){
            //	printf("%d\n", colMajored_sparse_idx[i]);
            //}
            //printf("\n");
        }
        //for (int i = 0; i < rows + 1; ++i){
        //	printf("R%d  C%d \n", row_ptr[i], col_ptr[i]);
        //}
        //printf("\n");
        //for (int i = 0; i < nnz; ++i){
        //	printf("%d %d %.3f\n", row_idx[i], col_idx[i], val[i]);
        //}

    }

    long nnz_of_row(long i) const { return (row_ptr[i + 1] - row_ptr[i]); }

    long nnz_of_col(long i) const { return (col_ptr[i + 1] - col_ptr[i]); }

    float get_global_mean() {
        float sum = 0;
        for (long i = 0; i < nnz; ++i) { sum += val[i]; }
        return sum / nnz;
    }

    ~smat_t() {
        if (mem_alloc_by_me) {
            //puts("Warnning: Somebody just free me.");
            free(val);
            free(val_t);
            free(row_ptr);
            free(row_idx);
            free(col_ptr);
            free(col_idx);
        }
    }

    smat_t transpose() {
        smat_t mt;
        mt.cols = rows;
        mt.rows = cols;
        mt.nnz = nnz;
        mt.val = val_t;
        mt.val_t = val;
        mt.nbits_val = nbits_val_t;
        mt.nbits_val_t = nbits_val;
        mt.col_ptr = row_ptr;
        mt.row_ptr = col_ptr;
        mt.nbits_col_ptr = nbits_row_ptr;
        mt.nbits_row_ptr = nbits_col_ptr;
        mt.col_idx = row_idx;
        mt.row_idx = col_idx;
        mt.nbits_col_idx = nbits_row_idx;
        mt.nbits_row_idx = nbits_col_idx;
        mt.max_col_nnz = max_row_nnz;
        mt.max_row_nnz = max_col_nnz;
        return mt;
    }
};

// Test set in COO format
class testset_t {
public:
    long rows;
    long cols;
    long nnz;
    long* test_row;
    long* test_col;
    float* test_val;

    void load(long _rows, long _cols, long _nnz, const char* filename) {
        long r;
        long c;
        float v;
        rows = _rows;
        cols = _cols;
        nnz = _nnz;

        test_row = new long[nnz];
        test_col = new long[nnz];
        test_val = new float[nnz];

        FILE* fp = fopen(filename, "r");
        for (long idx = 0; idx < nnz; ++idx) {
            fscanf(fp, "%ld %ld %f", &r, &c, &v);
            test_row[idx] = r - 1;
            test_col[idx] = c - 1;
            test_val[idx] = v;
        }
        fclose(fp);
    }

    ~testset_t() {
        delete[] test_row;
        delete[] test_col;
        delete[] test_val;
    }

};

#endif
