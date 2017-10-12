#ifndef MATUTILORIGINAL
#define MATUTILORIGINAL
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <cmath>
#include <omp.h>
#include <assert.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

enum { ROWMAJOR_Double, COLMAJOR_Double };

using namespace std;
class rate_t_Double;
class rateset_t_Double;
class RateComp_Double;
class smat_t_Double;
class testset_t_Double;
typedef vector<double> vec_t_Double;
typedef vector<vec_t_Double> mat_t_Double;
void load(const char* srcdir, smat_t_Double &R, testset_t_Double &T, bool with_weights = false);
void save_mat_t(mat_t_Double A, FILE *fp, bool row_major = true);
mat_t_Double load_mat_t_Double(FILE *fp, bool row_major = true);
void initial(mat_t_Double &X, long n, long k);
void initial_col(mat_t_Double &X, long k, long n);
double dot(const vec_t_Double &a, const vec_t_Double &b);
double dot(const mat_t_Double &W, const int i, const mat_t_Double &H, const int j);
double dot(const mat_t_Double &W, const int i, const vec_t_Double &H_j);
double norm(const vec_t_Double &a);
double norm(const mat_t_Double &M);
double calloss(const smat_t_Double &R, const mat_t_Double &W, const mat_t_Double &H);
double calobj(const smat_t_Double &R, const mat_t_Double &W, const mat_t_Double &H, const double lambda, bool iscol = false);

double calrmse(testset_t_Double &testset, const mat_t_Double &W, const mat_t_Double &H, bool iscol = false);
double calrmse_r1(testset_t_Double &testset, vec_t_Double &Wt, vec_t_Double &H_t);
double calrmse_r1(testset_t_Double &testset, vec_t_Double &Wt, vec_t_Double &Ht, vec_t_Double &oldWt, vec_t_Double &oldHt);

class rate_t_Double{
public:
	int i, j; double v, weight;
	rate_t_Double(int ii = 0, int jj = 0, double vv = 0, double ww = 1.0) : i(ii), j(jj), v(vv), weight(ww){}
};

class entry_iterator_t_Double{
private:
	FILE *fp;
	char buf[1000];
public:
	bool with_weights;
	size_t nnz;
	entry_iterator_t_Double() :nnz(0), fp(NULL), with_weights(false){}
	entry_iterator_t_Double(size_t nnz_, const char* filename, bool with_weights_ = false) {
		nnz = nnz_;
		fp = fopen(filename, "r");
		with_weights = with_weights_;
	}
	size_t size() { return nnz; }
	virtual rate_t_Double next() {
		int i = 1, j = 1;
		double v = 0, w = 1.0;
		if (nnz > 0) {
			fgets(buf, 1000, fp);
			if (with_weights)
				sscanf(buf, "%d %d %lf %lf", &i, &j, &v, &w);
			else
				sscanf(buf, "%d %d %lf", &i, &j, &v);
			--nnz;
		}
		else {
			fprintf(stderr, "Error: no more entry to iterate !!\n");
		}
		return rate_t_Double(i - 1, j - 1, v, w);
	}
	virtual ~entry_iterator_t_Double(){
		if (fp) fclose(fp);
	}
};



// Comparator for sorting rates into row/column comopression storage
class SparseComp_Double {
public:
	const unsigned *row_idx;
	const unsigned *col_idx;
	SparseComp_Double(const unsigned *row_idx_, const unsigned *col_idx_, bool isRCS_ = true) {
		row_idx = (isRCS_) ? row_idx_ : col_idx_;
		col_idx = (isRCS_) ? col_idx_ : row_idx_;
	}
	bool operator()(size_t x, size_t y) const {
		return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x] <= col_idx[y]));
	}
};

// Sparse matrix format CCS & RCS
// Access column fomat only when you use it..
class smat_t_Double{
public:
	long rows, cols;
	long nnz, max_row_nnz, max_col_nnz;
	double *val, *val_t;
	double *weight, *weight_t;
	long *col_ptr, *row_ptr;
	long *col_nnz, *row_nnz;
	unsigned *row_idx, *col_idx;    // condensed
	//unsigned long *row_idx, *col_idx; // for matlab
	bool mem_alloc_by_me, with_weights;
	smat_t_Double() :mem_alloc_by_me(false), with_weights(false){ }
	smat_t_Double(const smat_t_Double& m){ *this = m; mem_alloc_by_me = false; }

	// For matlab (Almost deprecated)
	smat_t_Double(long m, long n, unsigned *ir, long *jc, double *v, unsigned *ir_t, long *jc_t, double *v_t) :
		//smat_t_Double(long m, long n, unsigned long *ir, long *jc, double *v, unsigned long *ir_t, long *jc_t, double *v_t):
		rows(m), cols(n), mem_alloc_by_me(false),
		row_idx(ir), col_ptr(jc), val(v), col_idx(ir_t), row_ptr(jc_t), val_t(v_t) {
		if (col_ptr[n] != row_ptr[m])
			fprintf(stderr, "Error occurs! two nnz do not match (%ld, %ld)\n", col_ptr[n], row_ptr[n]);
		nnz = col_ptr[n];
		max_row_nnz = max_col_nnz = 0;
		for (long r = 1; r <= rows; ++r)
			max_row_nnz = max(max_row_nnz, row_ptr[r]);
		for (long c = 1; c <= cols; ++c)
			max_col_nnz = max(max_col_nnz, col_ptr[c]);
	}

	void from_mpi(){
		mem_alloc_by_me = true;
		max_col_nnz = 0;
		for (long c = 1; c <= cols; ++c)
			max_col_nnz = max(max_col_nnz, col_ptr[c] - col_ptr[c - 1]);
	}
	void print_mat(int host){
		for (int c = 0; c < cols; ++c) if (col_ptr[c + 1]>col_ptr[c]){
			printf("%d: %ld at host %d\n", c, col_ptr[c + 1] - col_ptr[c], host);
		}
	}
	void load(long _rows, long _cols, long _nnz, const char* filename, bool with_weights = false){
		entry_iterator_t_Double entry_it(_nnz, filename, with_weights);
		load_from_iterator(_rows, _cols, _nnz, &entry_it);
	}
	void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t_Double* entry_it) {
		rows = _rows, cols = _cols, nnz = _nnz;
		mem_alloc_by_me = true;
		with_weights = entry_it->with_weights;
		val = MALLOC(double, nnz); val_t = MALLOC(double, nnz);
		if (with_weights) { weight = MALLOC(double, nnz); weight_t = MALLOC(double, nnz); }
		row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);  // switch to this for memory
		row_ptr = MALLOC(long, rows + 1); col_ptr = MALLOC(long, cols + 1);
		memset(row_ptr, 0, sizeof(long)*(rows + 1));
		memset(col_ptr, 0, sizeof(long)*(cols + 1));

		/*
		* Assume ratings are stored in the row-majored ordering
		for(size_t idx = 0; idx < _nnz; idx++){
		rate_t_Double rate = entry_it->next();
		row_ptr[rate.i+1]++;
		col_ptr[rate.j+1]++;
		col_idx[idx] = rate.j;
		val_t[idx] = rate.v;
		}*/

		// a trick here to utilize the space the have been allocated 
		vector<size_t> perm(_nnz);
		unsigned *tmp_row_idx = col_idx;
		unsigned *tmp_col_idx = row_idx;
		double *tmp_val = val;
		double *tmp_weight = weight;
		for (size_t idx = 0; idx < _nnz; idx++){
			rate_t_Double rate = entry_it->next();
			row_ptr[rate.i + 1]++;
			col_ptr[rate.j + 1]++;
			tmp_row_idx[idx] = rate.i;
			tmp_col_idx[idx] = rate.j;
			tmp_val[idx] = rate.v;
			if (with_weights)
				tmp_weight[idx] = rate.weight;
			perm[idx] = idx;
		}
		// sort entries into row-majored ordering
		sort(perm.begin(), perm.end(), SparseComp_Double(tmp_row_idx, tmp_col_idx, true));
		// Generate CRS format
		for (size_t idx = 0; idx < _nnz; idx++) {
			val_t[idx] = tmp_val[perm[idx]];
			col_idx[idx] = tmp_col_idx[perm[idx]];
			if (with_weights)
				weight_t[idx] = tmp_weight[idx];
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
		for (long r = 0; r<rows; ++r){
			for (long i = row_ptr[r]; i < row_ptr[r + 1]; ++i){
				long c = col_idx[i];
				row_idx[col_ptr[c]] = r;
				val[col_ptr[c]] = val_t[i];
				if (with_weights) weight[col_ptr[c]] = weight_t[i];
				col_ptr[c]++;
			}
		}
		for (long c = cols; c>0; --c) col_ptr[c] = col_ptr[c - 1];
		col_ptr[0] = 0;
	}
	long nnz_of_row(int i) const { return (row_ptr[i + 1] - row_ptr[i]); }
	long nnz_of_col(int i) const { return (col_ptr[i + 1] - col_ptr[i]); }
	double get_global_mean(){
		double sum = 0;
		for (long i = 0; i<nnz; ++i) sum += val[i];
		return sum / nnz;
	}
	void remove_bias(double bias = 0){
		if (bias) {
			for (long i = 0; i<nnz; ++i) val[i] -= bias;
			for (long i = 0; i<nnz; ++i) val_t[i] -= bias;
		}
	}
	void free(void *ptr) { if (ptr) ::free(ptr); }
	~smat_t_Double(){
		if (mem_alloc_by_me) {
			//puts("Warnning: Somebody just free me.");
			free(val); free(val_t);
			free(row_ptr); free(row_idx);
			free(col_ptr); free(col_idx);
			if (with_weights) { free(weight); free(weight_t); }
		}
	}
	void clear_space() {
		free(val); free(val_t);
		free(row_ptr); free(row_idx);
		free(col_ptr); free(col_idx);
		if (with_weights) { free(weight); free(weight_t); }
		mem_alloc_by_me = false;
		with_weights = false;

	}
	smat_t_Double transpose(){
		smat_t_Double mt;
		mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
		mt.val = val_t; mt.val_t = val;
		mt.with_weights = with_weights;
		mt.weight = weight_t; mt.weight_t = weight;
		mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
		mt.col_idx = row_idx; mt.row_idx = col_idx;
		mt.max_col_nnz = max_row_nnz; mt.max_row_nnz = max_col_nnz;
		return mt;
	}
};


// row-major iterator
class smat_iterator_t_Double : public entry_iterator_t_Double{
private:
	unsigned *col_idx;
	long *row_ptr;
	double *val_t;
	double *weight_t;
	size_t	rows, cols, cur_idx, cur_row;
	bool with_weights;
public:
	smat_iterator_t_Double(const smat_t_Double& M, int major = ROWMAJOR_Double) {
		nnz = M.nnz;
		col_idx = (major == ROWMAJOR_Double) ? M.col_idx : M.row_idx;
		row_ptr = (major == ROWMAJOR_Double) ? M.row_ptr : M.col_ptr;
		val_t = (major == ROWMAJOR_Double) ? M.val_t : M.val;
		weight_t = (major == ROWMAJOR_Double) ? M.weight_t : M.weight;
		with_weights = M.with_weights;
		rows = (major == ROWMAJOR_Double) ? M.rows : M.cols;
		cols = (major == ROWMAJOR_Double) ? M.cols : M.rows;
		cur_idx = cur_row = 0;
	}
	~smat_iterator_t_Double() {}
	rate_t_Double next() {
		int i = 1, j = 1;
		double v = 0;
		while (cur_idx >= row_ptr[cur_row + 1]) ++cur_row;
		if (nnz > 0) --nnz;
		else fprintf(stderr, "Error: no more entry to iterate !!\n");
		rate_t_Double ret(cur_row, col_idx[cur_idx], val_t[cur_idx], with_weights ? weight_t[cur_idx] : 1.0);
		cur_idx++;
		return ret;
	}
};


// Test set format
class testset_t_Double{
public:
	long rows, cols, nnz;
	vector<rate_t_Double> T;
	testset_t_Double() : rows(0), cols(0), nnz(0){}
	inline rate_t_Double& operator[](const unsigned &idx) { return T[idx]; }
	void load(long _rows, long _cols, long _nnz, const char *filename) {
		int r, c;
		double v;
		rows = _rows; cols = _cols; nnz = _nnz;
		T = vector<rate_t_Double>(nnz);
		FILE *fp = fopen(filename, "r");
		for (long idx = 0; idx < nnz; ++idx){
			fscanf(fp, "%d %d %lg", &r, &c, &v);
			T[idx] = rate_t_Double(r - 1, c - 1, v);
		}
		fclose(fp);
	}
	void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t_Double* entry_it){
		rows = _rows, cols = _cols, nnz = _nnz;
		T = vector<rate_t_Double>(nnz);
		for (size_t idx = 0; idx < nnz; ++idx)
			T[idx] = entry_it->next();
	}
	double get_global_mean(){
		double sum = 0;
		for (long i = 0; i<nnz; ++i) sum += T[i].v;
		return sum / nnz;
	}
	void remove_bias(double bias = 0){
		if (bias) for (long i = 0; i<nnz; ++i) T[i].v -= bias;
	}
};

#endif
