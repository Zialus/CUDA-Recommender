#ifndef _UTIL_H
#define _UTIL_H

#include <iostream>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <omp.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define SIZEBITS(type, size) sizeof(type)*(size)

enum { ROWMAJOR, COLMAJOR };

using namespace std;
class rate_t;
class rateset_t;
class RateComp;
class smat_t;
class testset_t;

typedef vector<float> vec_t;
typedef vector<vec_t> mat_t;

void load(const char* srcdir, smat_t& R, testset_t& T, bool ifALS, bool with_weights = false);
void save_mat_t(mat_t A, FILE* fp, bool row_major = true);
mat_t load_mat_t(FILE* fp, bool row_major = true);
void initial(mat_t& X, long n, long k);
void initial_col(mat_t& X, long k, long n);
float dot(const vec_t& a, const vec_t& b);
double dot(const mat_t& W, const int i, const mat_t& H, const int j);
float dot(const mat_t& W, const int i, const vec_t& H_j);
float norm(const vec_t& a);
float norm(const mat_t& M);
float calloss(const smat_t& R, const mat_t& W, const mat_t& H);
float calobj(const smat_t& R, const mat_t& W, const mat_t& H, const float lambda, bool iscol = false);
double calrmse(testset_t& testset, const mat_t& W, const mat_t& H, bool ifALS, bool iscol = false);
float calrmse_r1(testset_t& testset, vec_t& Wt, vec_t& H_t);
float calrmse_r1(testset_t& testset, vec_t& Wt, vec_t& Ht, vec_t& oldWt, vec_t& oldHt);

class rate_t {
public:
    int i, j;
    float v, weight;

    rate_t(int ii = 0, int jj = 0, float vv = 0, float ww = 1.0) : i(ii), j(jj), v(vv), weight(ww) {}
};

class entry_iterator_t {
private:
    FILE* fp;
    char buf[1000];
public:
    bool with_weights;
    size_t nnz;

    entry_iterator_t() : fp(nullptr), with_weights(false), nnz(0) {}

    entry_iterator_t(size_t nnz_, const char* filename, bool with_weights_ = false) {
        nnz = nnz_;
        fp = fopen(filename, "r");
        with_weights = with_weights_;
    }

    size_t size() { return nnz; }

    virtual rate_t next() {
        int i = 1, j = 1;
        float v = 0, w = 1.0;
        if (nnz > 0) {
            fgets(buf, 1000, fp);
            if (with_weights) {
                sscanf(buf, "%d %d %f %f", &i, &j, &v, &w);
            } else {
                sscanf(buf, "%d %d %f", &i, &j, &v);
            }
            --nnz;
        } else {
            fprintf(stderr, "Error: no more entry to iterate !!\n");
        }
        return rate_t(i - 1, j - 1, v, w);
    }

    virtual ~entry_iterator_t() {
        if (fp) { fclose(fp); }
    }
};


// Comparator for sorting rates into row/column comopression storage
class SparseComp {
public:
    const unsigned* row_idx;
    const unsigned* col_idx;

    SparseComp(const unsigned* row_idx_, const unsigned* col_idx_, bool isRCS_ = true) {
        row_idx = (isRCS_) ? row_idx_ : col_idx_;
        col_idx = (isRCS_) ? col_idx_ : row_idx_;
    }

    bool operator()(size_t x, size_t y) const {
        return (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x] <= col_idx[y]));
    }
};

// Sparse matrix format CCS & RCS
// Access column fomat only when you use it..
class smat_t {
public:
    long rows, cols;
    long nnz, max_row_nnz, max_col_nnz;
    float* val, * val_t;
    size_t nbits_val, nbits_val_t;
    float* weight, * weight_t;
    size_t nbits_weight, nbits_weight_t;
    long* col_ptr, * row_ptr;
    size_t nbits_col_ptr, nbits_row_ptr;
    long* col_nnz, * row_nnz;
    size_t nbits_col_nnz, nbits_row_nnz;
    unsigned* row_idx, * col_idx;    // condensed
    size_t nbits_row_idx, nbits_col_idx;
    unsigned* colMajored_sparse_idx;
    size_t nbits_colMajored_sparse_idx;
    //unsigned long *row_idx, *col_idx; // for matlab
    bool mem_alloc_by_me, with_weights;

    smat_t() : mem_alloc_by_me(false), with_weights(false) {}

    smat_t(const smat_t& m) {
        *this = m;
        mem_alloc_by_me = false;
    }

    void print_mat(int host) {
        for (int c = 0; c < cols; ++c) {
            if (col_ptr[c + 1] > col_ptr[c]) {
                printf("%d: %ld at host %d\n", c, col_ptr[c + 1] - col_ptr[c], host);
            }
        }
    }

    void load(long _rows, long _cols, long _nnz, const char* filename, bool ifALS, bool use_weights = false) {
        entry_iterator_t entry_it(_nnz, filename, use_weights);
        load_from_iterator(_rows, _cols, _nnz, &entry_it, ifALS);
    }

    void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t* entry_it, bool ifALS) {
        unsigned* mapIDX;
        rows = _rows, cols = _cols, nnz = _nnz;
        mem_alloc_by_me = true;
        with_weights = entry_it->with_weights;
        val = MALLOC(float, nnz);
        val_t = MALLOC(float, nnz);
        nbits_val = SIZEBITS(float, nnz);
        nbits_val_t = SIZEBITS(float, nnz);
        if (with_weights) {
            weight = MALLOC(float, nnz);
            weight_t = MALLOC(float, nnz);
            nbits_weight = SIZEBITS(float, nnz);
            nbits_weight_t = SIZEBITS(float, nnz);
        }
        row_idx = MALLOC(unsigned, nnz);
        col_idx = MALLOC(unsigned, nnz);  // switch to this for memory
        nbits_row_idx = SIZEBITS(unsigned, nnz);
        nbits_col_idx = SIZEBITS(unsigned, nnz);
        row_ptr = MALLOC(long, rows + 1);
        col_ptr = MALLOC(long, cols + 1);
        nbits_row_ptr = SIZEBITS(long, rows + 1);
        nbits_col_ptr = SIZEBITS(long, cols + 1);
        memset(row_ptr, 0, sizeof(long) * (rows + 1));
        memset(col_ptr, 0, sizeof(long) * (cols + 1));
        if (ifALS) {
            colMajored_sparse_idx = MALLOC(unsigned, nnz);
            nbits_colMajored_sparse_idx = SIZEBITS(unsigned, nnz);
        }
        /*
         * Assume ratings are stored in the row-majored ordering
        for(size_t idx = 0; idx < _nnz; idx++){
            rate_t rate = entry_it->next();
            row_ptr[rate.i+1]++;
            col_ptr[rate.j+1]++;
            col_idx[idx] = rate.j;
            val_t[idx] = rate.v;
        }*/

        // a trick here to utilize the space the have been allocated
        vector<size_t> perm(_nnz);
        unsigned* tmp_row_idx = col_idx;
        unsigned* tmp_col_idx = row_idx;
        float* tmp_val = val;
        float* tmp_weight = weight;
        for (size_t idx = 0; idx < _nnz; idx++) {
            rate_t rate = entry_it->next();
            row_ptr[rate.i + 1]++;
            col_ptr[rate.j + 1]++;
            tmp_row_idx[idx] = rate.i;
            tmp_col_idx[idx] = rate.j;
            tmp_val[idx] = rate.v;
            if (with_weights) {
                tmp_weight[idx] = rate.weight;
            }
            perm[idx] = idx;
        }
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
            if (with_weights) {
                weight_t[idx] = tmp_weight[idx];
            }
        }

        // Calculate nnz for each row and col
        max_row_nnz = max_col_nnz = 0;
        for (long r = 1; r <= rows; ++r) {
            max_row_nnz = max(max_row_nnz, row_ptr[r]);
            row_ptr[r] += row_ptr[r - 1];
        }
        for (long c = 1; c <= cols; ++c) {
            max_col_nnz = max(max_col_nnz, col_ptr[c]);
            col_ptr[c] += col_ptr[c - 1];
        }
        // Transpose CRS into CCS matrix
        for (long r = 0; r < rows; ++r) {
            for (long i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
                long c = col_idx[i];
                row_idx[col_ptr[c]] = r;
                val[col_ptr[c]] = val_t[i];
                if (with_weights) { weight[col_ptr[c]] = weight_t[i]; }
                col_ptr[c]++;
            }
        }
        for (long c = cols; c > 0; --c) { col_ptr[c] = col_ptr[c - 1]; }
        col_ptr[0] = 0;

        if (ifALS) {
            mapIDX = MALLOC(unsigned, rows);
            for (int r = 0; r < rows; ++r) {
                mapIDX[r] = row_ptr[r];
            }

            for (int r = 0; r < nnz; ++r) {
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

    long nnz_of_row(int i) const { return (row_ptr[i + 1] - row_ptr[i]); }

    long nnz_of_col(int i) const { return (col_ptr[i + 1] - col_ptr[i]); }

    float get_global_mean() {
        float sum = 0;
        for (long i = 0; i < nnz; ++i) { sum += val[i]; }
        return sum / nnz;
    }

    void remove_bias(float bias = 0) {
        if (bias) {
            for (long i = 0; i < nnz; ++i) { val[i] -= bias; }
            for (long i = 0; i < nnz; ++i) { val_t[i] -= bias; }
        }
    }

    void free(void* ptr) { if (ptr) { ::free(ptr); }}

    ~smat_t() {
        if (mem_alloc_by_me) {
            //puts("Warnning: Somebody just free me.");
            free(val);
            free(val_t);
            free(row_ptr);
            free(row_idx);
            free(col_ptr);
            free(col_idx);
            if (with_weights) {
                free(weight);
                free(weight_t);
            }
        }
    }

    void clear_space() {
        free(val);
        free(val_t);
        free(row_ptr);
        free(row_idx);
        free(col_ptr);
        free(col_idx);
        if (with_weights) {
            free(weight);
            free(weight_t);
        }
        mem_alloc_by_me = false;
        with_weights = false;

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
        mt.with_weights = with_weights;

        mt.weight = weight_t;
        mt.weight_t = weight;
        mt.nbits_weight = nbits_weight_t;
        mt.nbits_weight_t = nbits_weight;
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


// row-major iterator
class smat_iterator_t : public entry_iterator_t {
private:
    unsigned* col_idx;
    long* row_ptr;
    float* val_t;
    float* weight_t;
    size_t rows, cols, cur_idx, cur_row;
    bool with_weights;
public:
    smat_iterator_t(const smat_t& M, int major = ROWMAJOR) {
        nnz = M.nnz;
        col_idx = (major == ROWMAJOR) ? M.col_idx : M.row_idx;
        row_ptr = (major == ROWMAJOR) ? M.row_ptr : M.col_ptr;
        val_t = (major == ROWMAJOR) ? M.val_t : M.val;
        weight_t = (major == ROWMAJOR) ? M.weight_t : M.weight;
        with_weights = M.with_weights;
        rows = (major == ROWMAJOR) ? M.rows : M.cols;
        cols = (major == ROWMAJOR) ? M.cols : M.rows;
        cur_idx = cur_row = 0;
    }

    ~smat_iterator_t() {}

    rate_t next() {
//        int i = 1, j = 1;
//        float v = 0;
        while (cur_idx >= row_ptr[cur_row + 1]) { ++cur_row; }
        if (nnz > 0) { --nnz; }
        else { fprintf(stderr, "Error: no more entry to iterate !!\n"); }
        rate_t ret(cur_row, col_idx[cur_idx], val_t[cur_idx], with_weights ? weight_t[cur_idx] : 1.0f);
        cur_idx++;
        return ret;
    }
};


// Test set format
class testset_t {
public:
    long rows, cols, nnz;
    vector<rate_t> T;

    testset_t() : rows(0), cols(0), nnz(0) {}

    inline rate_t& operator[](const unsigned& idx) { return T[idx]; }

    void load(long _rows, long _cols, long _nnz, const char* filename) {
        int r, c;
        float v;
        rows = _rows;
        cols = _cols;
        nnz = _nnz;
        T = vector<rate_t>(nnz);
        FILE* fp = fopen(filename, "r");
        for (long idx = 0; idx < nnz; ++idx) {
            fscanf(fp, "%d %d %f", &r, &c, &v);
            T[idx] = rate_t(r - 1, c - 1, v);
        }
        fclose(fp);
    }

    void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t* entry_it) {
        rows = _rows, cols = _cols, nnz = _nnz;
        T = vector<rate_t>(nnz);
        for (size_t idx = 0; idx < nnz; ++idx) {
            T[idx] = entry_it->next();
        }
    }

    float get_global_mean() {
        float sum = 0;
        for (long i = 0; i < nnz; ++i) { sum += T[i].v; }
        return sum / nnz;
    }

    void remove_bias(float bias = 0) {
        if (bias) { for (long i = 0; i < nnz; ++i) { T[i].v -= bias; }}
    }
};

#endif
