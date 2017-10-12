#ifndef CCD_C_HEADER
#define CCD_C_HEADER

extern "C" {

	struct params {
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
	struct smat_t_C {
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
	};

	smat_t_C transpose(smat_t_C m);
	long nnz_of_row(int i, const long *row_ptr);
	long nnz_of_col(int i, const long *col_ptr);
	float maxC(float a, float b);
}

#endif