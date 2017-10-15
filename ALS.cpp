#include "pmf.h"
#include "ALS_headerToC.h"

#define kind dynamic,500
extern "C"  void kernel_wrapper_als_NV(smat_t_C_als &R_C, float ** &W, float ** &H, params_als &parameters);

void choldc1(int n, float** a, float* p) {
	unsigned i, j;
	int k;
	float sum;
	for (i = 0; i < n; ++i) {
		for (j = i; j < n; ++j) {
			sum = a[i][j];
			for (k = i - 1; k >= 0; --k) {
				sum -= a[i][k] * a[j][k];
			}
			if (i == j) {
				if (sum <= 0) {
					printf(" a is not positive definite!\n");
				}
				p[i] = sqrtf(sum);//float saquare root
			}
			else {
				a[j][i] = sum / p[i];
			}
		}
	}
}

void choldcsl(int n, float** A) {
	unsigned i, j, k; double sum;
	float* p;
	p = (float *)malloc(n * sizeof(float));
	choldc1(n, A, p);
	for (i = 0; i < n; ++i) {
		A[i][i] = 1 / p[i];
		for (j = i + 1; j < n; ++j) {
			sum = 0;
			for (k = i; k < j; ++k) {
				sum -= A[j][k] * A[k][i];
			}
			A[j][i] = sum / p[j];
		}
	}
	free(p);
}

void inverseMatrix_CholeskyMethod(int n, float** A) {
	unsigned i, j, k;
	choldcsl(n, A);
	//vecIndex = (i * 3) + j; to ontain index from vector if needed.
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {
			A[i][j] = 0.0;
		}
	}
	for (i = 0; i < n; i++) {
		A[i][i] *= A[i][i];
		for (k = i + 1; k < n; ++k) {
			A[i][i] += A[k][i] * A[k][i];
		}
		for (j = i + 1; j < n; ++j) {
			for (k = j; k < n; ++k) {
				A[i][j] += A[k][i] * A[k][j];
			}
		}
	}
	for (i = 0; i < n; ++i) {
		for (j = 0; j < i; ++j) {
			A[i][j] = A[j][i];
		}
	}
}


//Multiply matrix M transpose by M 
void Mt_byM_multiply(int i, int j, float**M, float**Result){
	float SUM;
	for (unsigned I = 0; I < j; ++I){
		for (unsigned J = I; J < j; ++J) {
			SUM = 0.0f;
			for (unsigned K = 0; K < i; ++K){
				//printf("%.3f %.3f\n", M[K][I], M[K][J]);
				SUM += M[K][I] * M[K][J];
			}
			Result[J][I] = SUM;
			Result[I][J] = SUM;
		}
	}
}
//void Mt_byM_multiply(int i, int j, float**M, float**Result){
//	float SUM;
//	for (unsigned I = 0; I < j; ++I){
//		for (unsigned J = 0; J < j; ++J) {
//			SUM = 0.0;
//			for (unsigned K = 0; K < i; ++K){
//				//printf("%.3f %.3f\n", M[K][I], M[K][J]);
//				SUM += M[K][I] * M[K][J];
//			}
//			Result[J][I] = SUM;
//		}
//	}
//
//}
//Multiply matrix M by M tranpose
void M_byMt_multiply(int i, int j, float**M, float**Result){
	float SUM;
	for (unsigned I = 0; I < i; ++I){
		for (unsigned J = 0; J < i; ++J) {
			SUM = 0.0;
			for (unsigned K = 0; K < j; ++K)
				SUM += M[I][K] * M[J][K];
			Result[I][J] = SUM;
		}
	}

}

void ALS(smat_t &R, mat_t &W, mat_t &H, testset_t &T, parameter &param){
	if (param.enable_cuda){
		printf("CUDA enabled version.\n");

		params_als parameters;
		parameters.k = param.k;
		parameters.maxiter = param.maxiter;
		parameters.inneriter = param.maxinneriter;
		parameters.lambda = param.lambda;
		parameters.do_nmf = param.do_nmf;
		parameters.verbose = param.verbose;
		parameters.enable_cuda = param.enable_cuda;
		parameters.nBlocks = param.nBlocks;
		parameters.nThreadsPerBlock = param.nThreadsPerBlock;

		smat_t_C_als R_C;
		R_C.cols = R.cols; R_C.rows = R.rows; R_C.nnz = R.nnz;
		R_C.val = R.val; R_C.val_t = R.val_t;
		R_C.nbits_val = R.nbits_val; R_C.nbits_val_t = R.nbits_val_t;
		R_C.with_weights = R.with_weights;
		R_C.weight = R.weight; R_C.weight_t = R.weight_t;
		R_C.nbits_weight = R.nbits_weight; R_C.nbits_weight_t = R.nbits_weight_t;
		R_C.col_ptr = R.col_ptr; R_C.row_ptr = R.row_ptr;
		R_C.nbits_col_ptr = R.nbits_col_ptr; R_C.nbits_row_ptr = R.nbits_row_ptr;
		R_C.col_idx = R.col_idx; R_C.row_idx = R.row_idx;
		R_C.nbits_col_idx = R.nbits_col_idx; R_C.nbits_row_idx = R.nbits_row_idx;
		R_C.max_col_nnz = R.max_col_nnz; R_C.max_row_nnz = R.max_row_nnz;
		R_C.colMajored_sparse_idx = R.colMajored_sparse_idx; R_C.nbits_colMajored_sparse_idx = R.nbits_colMajored_sparse_idx;

		float **W_c;
		float **H_c;

		H_c = (float **)malloc(R.cols * sizeof(float *));
		if (H_c == nullptr){
			fprintf(stderr, "out of memory\n");
			return;
		}
		for (int i = 0; i < R.cols; i++){
			H_c[i] = &H[i][0];
			if (H_c[i] == nullptr){
				fprintf(stderr, "out of memory\n");
				return;
			}
		}

		W_c = (float **)malloc(R.rows * sizeof(float *));
		if (W_c == nullptr){
			fprintf(stderr, "out of memory\n");
			return;
		}
		for (int i = 0; i < R.rows; i++){
			W_c[i] = &W[i][0];
			if (W_c[i] == nullptr){
				fprintf(stderr, "out of memory\n");
				return;
			}
		}

		kernel_wrapper_als_NV(R_C, W_c, H_c, parameters);

		free(W_c);
		free(H_c);
	}
	else{
		ALS_multicore(R, W, H, param);
	}
}


void ALS_multicore(smat_t &R, mat_t &W, mat_t &H, parameter &param){
	int maxIter = param.maxiter;
	float lambda = param.lambda;
	float k = param.k;
	int num_threads_old = omp_get_num_threads();

	omp_set_num_threads(param.threads);

	for (unsigned int iter = 0; iter < maxIter; ++iter){
		
		//optimize W over H
#pragma omp parallel for schedule(kind)
		for (int Rw = 0; Rw < R.rows; ++Rw){
			float *Wr = &W[Rw][0];
			unsigned omegaSize = R.row_ptr[Rw + 1] - R.row_ptr[Rw];
			float ** subMatrix, ** H_Omega;
			float * subVector;			

			if (omegaSize>0){
				subVector = (float *)malloc(k * sizeof(float));
				subMatrix = (float **)malloc(k * sizeof(float *));
				for (unsigned i = 0; i < k; ++i){
					subMatrix[i] = (float *)malloc(k * sizeof(float));
				}

				//a trick to avoid malloc
				H_Omega = (float **)malloc(omegaSize * sizeof(float *));
				unsigned i = 0;
				for (unsigned idx = R.row_ptr[Rw]; idx < R.row_ptr[Rw + 1]; ++idx){
					H_Omega[i] = &H[R.col_idx[idx]][0];
					++i;
				}

				Mt_byM_multiply(omegaSize, k, H_Omega, subMatrix);

				//add lambda to diag of sub-matrix
				for (unsigned c = 0; c < k; c++){
					subMatrix[c][c] = subMatrix[c][c] + lambda;
				}

				//invert sub-matrix
				inverseMatrix_CholeskyMethod(k, subMatrix);


				//sparse multiplication
				for (unsigned c=0; c < k; ++c){
					subVector[c] = 0;
					for (unsigned idx = R.row_ptr[Rw]; idx < R.row_ptr[Rw + 1]; ++idx){
						unsigned idx2 = R.colMajored_sparse_idx[idx];
						subVector[c] += R.val[idx2] * H[R.col_idx[idx]][c];
					}
				}

				//multiply subVector by subMatrix
				for (unsigned c = 0; c < k; ++c){
					Wr[c] = 0;
					for (unsigned subVid = 0; subVid < k; ++subVid){
						Wr[c] += subVector[subVid] * subMatrix[c][subVid];
					}
				}


				for (unsigned i = 0; i < k; ++i){
					free(subMatrix[i]);
				}
				free(subMatrix);
				free(subVector);
				free(H_Omega);
			}
			else{
				for (unsigned c = 0; c < k; ++c){
					Wr[c] = 0.0f;
					//printf("%.3f ", Wr[c]);
				}
				//printf("\n");
			}		
		}

		//optimize H over W
#pragma omp parallel for schedule(kind)
		for (int Rh = 0; Rh < R.cols; ++Rh){
			float *Hr = &H[Rh][0];
			unsigned omegaSize = R.col_ptr[Rh + 1] - R.col_ptr[Rh];
			float ** subMatrix, ** W_Omega;
			float * subVector;

			if (omegaSize>0){
				subVector = (float *)malloc(k * sizeof(float));
				subMatrix = (float **)malloc(k * sizeof(float *));
				for (unsigned i = 0; i < k; ++i){
					subMatrix[i] = (float *)malloc(k * sizeof(float));
				}

				//a trick to avoid malloc
				W_Omega = (float **)malloc(omegaSize * sizeof(float *));
				unsigned i = 0;
				for (unsigned idx = R.col_ptr[Rh]; idx < R.col_ptr[Rh + 1]; ++idx){
					W_Omega[i] = &W[R.row_idx[idx]][0];
					++i;
				}

				Mt_byM_multiply(omegaSize, k, W_Omega, subMatrix);

				//add lambda to diag of sub-matrix
				for (unsigned c = 0; c < k; c++){
					subMatrix[c][c] = subMatrix[c][c] + lambda;
				}

				//invert sub-matrix
				inverseMatrix_CholeskyMethod(k, subMatrix);


				//sparse multiplication
				for (unsigned c = 0; c < k; ++c){
					subVector[c] = 0;
					for (unsigned idx = R.col_ptr[Rh]; idx < R.col_ptr[Rh + 1]; ++idx){
						subVector[c] += R.val[idx] * W[R.row_idx[idx]][c];
					}
				}

				//multiply subVector by subMatrix
				for (unsigned c = 0; c < k; ++c){
					Hr[c] = 0;
					for (unsigned subVid = 0; subVid < k; ++subVid){
						Hr[c] += subVector[subVid] * subMatrix[c][subVid];
					}
				}


				for (unsigned i = 0; i < k; ++i){
					free(subMatrix[i]);
				}
				free(subMatrix);
				free(subVector);
				free(W_Omega);
			}
			else{
				for (unsigned c = 0; c < k; ++c){
					Hr[c] = 0.0f;
				}
			}
		}

	}
	omp_set_num_threads(num_threads_old);
}