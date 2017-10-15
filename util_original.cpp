#include "util_original.h"


#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))


// load utility for CCS RCS
void load(const char* srcdir, smat_t_Double &R, testset_t_Double &T, bool with_weights){
	// add testing later
	char filename[1024], buf[1024];
	sprintf(filename, "%s/meta", srcdir);
	FILE *fp = fopen(filename, "r");
	long m, n, nnz;
	fscanf(fp, "%ld %ld", &m, &n);

	fscanf(fp, "%ld %s", &nnz, buf);
	sprintf(filename, "%s/%s", srcdir, buf);
	R.load(m, n, nnz, filename, with_weights);

	if (fscanf(fp, "%ld %s", &nnz, buf) != EOF){
		sprintf(filename, "%s/%s", srcdir, buf);
		T.load(m, n, nnz, filename);
	}
	fclose(fp);
	//double bias = R.get_global_mean(); R.remove_bias(bias); T.remove_bias(bias);
	return;
}

// Save a mat_t_Double A to a file in row_major order.
// row_major = true: A is stored in row_major order,
// row_major = false: A is stored in col_major order.
void save_mat_t(mat_t_Double A, FILE *fp, bool row_major){
	if (fp == NULL)
		fprintf(stderr, "output stream is not valid.\n");
	long m = row_major ? A.size() : A[0].size();
	long n = row_major ? A[0].size() : A.size();
	fwrite(&m, sizeof(long), 1, fp);
	fwrite(&n, sizeof(long), 1, fp);
	vec_t_Double buf(m*n);

	if (row_major) {
		size_t idx = 0;
		for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			buf[idx++] = A[i][j];
	}
	else {
		size_t idx = 0;
		for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			buf[idx++] = A[j][i];
	}
	fwrite(&buf[0], sizeof(double), m*n, fp);
}

// Load a matrix from a file and return a mat_t_Double matrix 
// row_major = true: the returned A is stored in row_major order,
// row_major = false: the returened A  is stored in col_major order.
mat_t_Double load_mat_t_Double(FILE *fp, bool row_major){
	if (fp == NULL)
		fprintf(stderr, "input stream is not valid.\n");
	long m, n;
	fread(&m, sizeof(long), 1, fp);
	fread(&n, sizeof(long), 1, fp);
	vec_t_Double buf(m*n);
	fread(&buf[0], sizeof(double), m*n, fp);
	mat_t_Double A;
	if (row_major) {
		A = mat_t_Double(m, vec_t_Double(n));
		size_t idx = 0;
		for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			A[i][j] = buf[idx++];
	}
	else {
		A = mat_t_Double(n, vec_t_Double(m));
		size_t idx = 0;
		for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			A[j][i] = buf[idx++];
	}
	return A;
}
void initial(mat_t_Double &X, long n, long k){
	X = mat_t_Double(n, vec_t_Double(k));
	srand(0L);//srand48(0L);//########################################################################################################################################
	for (long i = 0; i < n; ++i){
		for (long j = 0; j < k; ++j)
			X[i][j] = 0.1*(double(rand()) / RAND_MAX);//X[i][j] = 0.1*drand48(); //-1;//########################################################################################################################################
		//X[i][j] = 0; //-1;
	}
}

void initial_col(mat_t_Double &X, long k, long n){
	X = mat_t_Double(k, vec_t_Double(n));
	srand(0L);//srand48(0L);//########################################################################################################################################
	for (long i = 0; i < n; ++i)
	for (long j = 0; j < k; ++j)
		X[j][i] = 0.1*(double(rand()) / RAND_MAX);//X[j][i] = 0.1*drand48();
}

double dot(const vec_t_Double &a, const vec_t_Double &b){
	double ret = 0;
#pragma omp parallel for 
	for (int i = a.size() - 1; i >= 0; --i)
		ret += a[i] * b[i];
	return ret;
}
double dot(const mat_t_Double &W, const int i, const mat_t_Double &H, const int j){
	int k = W.size();
	double ret = 0;
	for (int t = 0; t < k; ++t)
		ret += W[t][i] * H[t][j];
	return ret;
}
double dot(const mat_t_Double &W, const int i, const vec_t_Double &H_j){
	int k = H_j.size();
	double ret = 0;
	for (int t = 0; t < k; ++t)
		ret += W[t][i] * H_j[t];
	return ret;
}
double norm(const vec_t_Double &a){
	double ret = 0;
	for (int i = a.size() - 1; i >= 0; --i)
		ret += a[i] * a[i];
	return ret;
}
double norm(const mat_t_Double &M) {
	double reg = 0;
	for (int i = M.size() - 1; i >= 0; --i) reg += norm(M[i]);
	return reg;
}
double calloss(const smat_t_Double &R, const mat_t_Double &W, const mat_t_Double &H){
	double loss = 0;
	int k = H.size();
	for (long c = 0; c < R.cols; ++c){
		for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx){
			double diff = -R.val[idx];
			diff += dot(W[R.row_idx[idx]], H[c]);
			loss += diff*diff;
		}
	}
	return loss;
}
double calobj(const smat_t_Double &R, const mat_t_Double &W, const mat_t_Double &H, const double lambda, bool iscol){
	double loss = 0;
	int k = iscol ? H.size() : 0;
	vec_t_Double Hc(k);
	for (long c = 0; c < R.cols; ++c){
		if (iscol)
		for (int t = 0; t<k; ++t) Hc[t] = H[t][c];
		for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx){
			double diff = -R.val[idx];
			if (iscol)
				diff += dot(W, R.row_idx[idx], Hc);
			else
				diff += dot(W[R.row_idx[idx]], H[c]);
			loss += (R.with_weights ? R.weight[idx] : 1.0) * diff*diff;
		}
	}
	double reg = 0;
	if (iscol) {
		for (int t = 0; t<k; ++t) {
			for (long r = 0; r<R.rows; ++r) reg += W[t][r] * W[t][r] * R.nnz_of_row(r);
			for (long c = 0; c<R.cols; ++c) reg += H[t][c] * H[t][c] * R.nnz_of_col(c);
		}
	}
	else {
		for (long r = 0; r<R.rows; ++r) reg += R.nnz_of_row(r)*norm(W[r]);
		for (long c = 0; c<R.cols; ++c) reg += R.nnz_of_col(c)*norm(H[c]);
	}
	reg *= lambda;
	return loss + reg;
}

double calrmse(testset_t_Double &testset, const mat_t_Double &W, const mat_t_Double &H, bool iscol){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
	for (size_t idx = 0; idx < nnz; ++idx){
		err = -testset[idx].v;
		if (iscol)
			err += dot(W, testset[idx].i, H, testset[idx].j);
		else
			err += dot(W[testset[idx].i], H[testset[idx].j]);
		rmse += err*err;
	}
	return sqrt(rmse / nnz);
}

double calrmse_r1(testset_t_Double &testset, vec_t_Double &Wt, vec_t_Double &Ht){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
	for (int idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i] * Ht[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse / nnz);
}

double calrmse_r1(testset_t_Double &testset, vec_t_Double &Wt, vec_t_Double &Ht, vec_t_Double &oldWt, vec_t_Double &oldHt){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
	for (int idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i] * Ht[testset[idx].j] - oldWt[testset[idx].i] * oldHt[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse / nnz);
}


