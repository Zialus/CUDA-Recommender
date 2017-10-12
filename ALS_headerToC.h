#ifndef ALS_C_HEADER
#define ALS_C_HEADER


struct params_als {
	int nBlocks;
	int nThreadsPerBlock;
	int k;
	int maxiter;
	int inneriter;
	int do_nmf;
	int verbose;
	float lambda;
	float eps;
	bool enable_cuda;
};
struct smat_t_C_als {
	long rows, cols;
	long nnz, max_row_nnz, max_col_nnz;
	float *val, *val_t;
	size_t nbits_val, nbits_val_t;
	float *weight, *weight_t;
	size_t nbits_weight, nbits_weight_t;
	long *col_ptr, *row_ptr;
	size_t nbits_col_ptr, nbits_row_ptr;
	//long *col_nnz, *row_nnz;
	size_t nbits_col_nnz, nbits_row_nnz;
	unsigned *row_idx, *col_idx;
	size_t nbits_row_idx, nbits_col_idx;
	bool with_weights;
	unsigned *colMajored_sparse_idx;
	size_t nbits_colMajored_sparse_idx;
};

void Mt_byM_multiply_(int i, int j, float**M, float**Result);
void inverseMatrix_CholeskyMethod_(int n, float** A);
void choldcsl_(int n, float** A);
void choldc1_(int n, float** a, float* p);

void Mt_byM_multiply_off(int i, int j, float*H, float**Result, const long ptr, const unsigned *idx);
void inverseMatrix_CholeskyMethod_off(int n, float* A);
void choldcsl_off(int n, float* A);
void choldc1_off(int n, float* a, float* p);

#endif