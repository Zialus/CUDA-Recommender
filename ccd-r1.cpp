#include "stdafx.h"//########################################################################################################################################

#include "pmf.h"
#include "ccd_headerToC.h"
//#include <cuda.h>

#define kind dynamic,500
extern "C"  void kernel_wrapper_ccdpp_NV(smat_t_C &R_C, float ** &W, float ** &H, params &parameters);

// CCD rank-one 
inline float RankOneUpdate_Original_float(const smat_t &R, const int j, const vec_t &u, const float lambda, const float vj, float *redvar, int do_nmf){
	float g=0, h=lambda;
	if(R.col_ptr[j+1]==R.col_ptr[j]) return 0;
	for(long idx=R.col_ptr[j]; idx < R.col_ptr[j+1]; ++idx) {
		int i = R.row_idx[idx];
		g += u[i]*R.val[idx]; 
		h += u[i]*u[i];
	}
	float newvj = g/h, tmp = 0, delta = 0, fundec = 0;
	if(do_nmf>0 & newvj < 0) {
		newvj = 0;
		delta = vj; // old - new
		fundec = -2*g*vj; //+ h*vj*vj;
	} else {
		delta = vj - newvj;
		fundec = h*delta*delta;
	}
	//float delta = vj - newvj;
	//float fundec = h*delta*delta;
	//float lossdec = fundec - lambda*delta*(vj+newvj);
	//float gnorm = (g-h*vj)*(g-h*vj); 
	*redvar += fundec;
	//*redvar += lossdec;
	return newvj;
}

inline float UpdateRating_Original_float(smat_t &R, const vec_t &Wt, const vec_t &Ht, const vec_t &oldWt, const vec_t &oldHt) {
	float loss=0;
#pragma omp parallel for  schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			float Htc = Ht[c], oldHtc = oldHt[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt[R.row_idx[idx]]*Htc-oldWt[R.row_idx[idx]]*oldHtc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
	int tt = 0;
	return loss;	
}
inline float UpdateRating_Original_float(smat_t &R, const vec_t &Wt2, const vec_t &Ht2) {
	float loss=0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			float Htc = Ht2[2*c], oldHtc = Ht2[2*c+1], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt2[2*R.row_idx[idx]]*Htc-Wt2[2*R.row_idx[idx]+1]*oldHtc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
	return loss;	
}

inline float UpdateRating_Original_float(smat_t &R, const vec_t &Wt, const vec_t &Ht, bool add) {
	float loss=0;
	if(add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			float Htc = Ht[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] +=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;	
	} else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			float Htc = Ht[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;	
	}
}




// Cyclic Coordinate Descent for Matrix Factorization
void ccdr1(smat_t &R, mat_t &W, mat_t &H, testset_t &T, parameter &param){

	if (param.enable_cuda){
		printf("CUDA enabled version.\n");

		//By refefrence version.
		params parameters;
		parameters.k = param.k;
		parameters.maxiter = param.maxiter;
		parameters.inneriter = param.maxinneriter;
		parameters.lambda = param.lambda;
		parameters.do_nmf = param.do_nmf;
		parameters.verbose = param.verbose;
		parameters.enable_cuda = param.enable_cuda;
		parameters.nBlocks = param.nBlocks;
		parameters.nThreadsPerBlock = param.nThreadsPerBlock;

		smat_t_C R_C;
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

		float **W_c;
		float **H_c;

		H_c = (float **)malloc(param.k * sizeof(float *));
		if (H_c == NULL){
			fprintf(stderr, "out of memory\n");
			return;
		}
		for (int i = 0; i < param.k; i++){
			H_c[i] = &H[i][0];
			if (H_c[i] == NULL){
				fprintf(stderr, "out of memory\n");
				return;
			}
		}

		W_c = (float **)malloc(param.k * sizeof(float *));
		if (W_c == NULL){
			fprintf(stderr, "out of memory\n");
			return;
		}
		for (int i = 0; i < param.k; i++){
			W_c[i] = &W[i][0];
			if (W_c[i] == NULL){
				fprintf(stderr, "out of memory\n");
				return;
			}
		}

		kernel_wrapper_ccdpp_NV(R_C, W_c, H_c, parameters);

		free(W_c);
		free(H_c);

	}
	else{
		ccdr1_original_float(R, W, H, T, param);
	}
}


// Cyclic Coordinate Descent for Matrix Factorization
void ccdr1_original_float(smat_t &R, mat_t &W, mat_t &H, testset_t &T, parameter &param){
	fprintf(stdout, "Float OMP Implementation\n");
	int k = param.k;
	int maxiter = param.maxiter;
	int inneriter = param.maxinneriter;
	int num_threads_old = omp_get_num_threads();
	float lambda = param.lambda;
	float eps = param.eps;
	float Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj=0;
	long num_updates = 0;
	float reg=0,loss;

	omp_set_num_threads(param.threads);

	// Create transpose view of R
	smat_t Rt;
	Rt = R.transpose();
	// initial value of the regularization term
	// H is a zero matrix now.
	for(int t=0;t<k;++t) for(long c=0;c<R.cols;++c) H[t][c] = 0; 
	for(int t=0;t<k;++t) for(long r=0;r<R.rows;++r) reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
	
	vec_t oldWt(R.rows), oldHt(R.cols);
	vec_t u(R.rows), v(R.cols);
	for(int oiter = 1; oiter <= maxiter; ++oiter) {
		float gnorm = 0, initgnorm=0;
		float rankfundec = 0;
		float fundec_max = 0;
		int early_stop = 0;
		for(int tt=0; tt < k; ++tt) {
			int t = tt;
			if(early_stop >= 5) break;
			//if(oiter>1) { t = rand()%k; }
			start = omp_get_wtime();
			vec_t &Wt = W[t], &Ht = H[t];
#pragma omp parallel for
			for(int i = 0; i < R.rows; ++i) oldWt[i] = u[i]= Wt[i];
#pragma omp parallel for
			for(int i = 0; i < R.cols; ++i) {v[i]= Ht[i]; oldHt[i] = (oiter == 1)? 0: v[i];}

			// Create Rhat = R - Wt Ht^T
			if (oiter > 1) {
				UpdateRating_Original_float(R, Wt, Ht, true);
				UpdateRating_Original_float(Rt, Ht, Wt, true);
			} 
			Itime += omp_get_wtime() - start;

			gnorm = 0, initgnorm=0;
			float innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter; 	
			//	if(oiter > 1) maxit *= 2;
			for(int iter = 1; iter <= maxit; ++iter){
				// Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				innerfundec_cur = 0;
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < R.cols; ++c)
					v[c] = RankOneUpdate_Original_float(R, c, u, lambda*(R.col_ptr[c+1]-R.col_ptr[c]), v[c], &innerfundec_cur, param.do_nmf);
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;
				// Update W[t]
				start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < Rt.cols; ++c)
					u[c] = RankOneUpdate_Original_float(Rt, c, v, lambda*(Rt.col_ptr[c + 1] - Rt.col_ptr[c]), u[c], &innerfundec_cur, param.do_nmf);
				num_updates += Rt.cols;
				if((innerfundec_cur < fundec_max*eps))  {
					if(iter==1) early_stop+=1;
					break; 
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = max(innerfundec_max, innerfundec_cur);
				// the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
				if(!(oiter==1 && t == 0 && iter==1))
					fundec_max = max(fundec_max, innerfundec_cur);
				Wtime += omp_get_wtime() - start;
			}

			// Update R and Rt
			start = omp_get_wtime();
#pragma omp parallel for
			for(int i = 0; i < R.rows; ++i) Wt[i]= u[i];
#pragma omp parallel for
			for(int i = 0; i < R.cols; ++i) Ht[i]= v[i];
			loss = UpdateRating_Original_float(R, u, v, false);
			loss = UpdateRating_Original_float(Rt, v, u, false);
			Rtime += omp_get_wtime() - start;

			for(long c = 0; c < R.cols; ++c) {
				reg += R.nnz_of_col(c)*Ht[c]*Ht[c];
				reg -= R.nnz_of_col(c)*oldHt[c]*oldHt[c];
			}
			for(long c = 0; c < Rt.cols; ++c) {
				reg += Rt.nnz_of_col(c)*(Wt[c]*Wt[c]);
				reg -= Rt.nnz_of_col(c)*(oldWt[c]*oldWt[c]);
			}
			float obj = loss+reg*lambda;
			if(param.verbose)
				printf("iter %d rank %d time %.10g loss %.10g obj %.10g diff %.10g gnorm %.6g reg %.7g ",
						oiter,t+1, Htime+Wtime+Rtime, loss, obj, oldobj - obj, initgnorm, reg);
			oldobj = obj;
			if(T.nnz!=0 && param.do_predict){ //if(T.nnz!=0 and param.do_predict){########################################################################################################################################
				if(param.verbose)
					printf("rmse %.10g", calrmse_r1(T, Wt, Ht, oldWt, oldHt)); 
			}
			if(param.verbose) puts("");
			fflush(stdout);
		}
	}
	omp_set_num_threads(num_threads_old);
}

