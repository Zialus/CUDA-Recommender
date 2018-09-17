#include "CCDPP_onCUDA.h"

__global__ void RankOneUpdate_DUAL_kernel(const long Rcols, //are the iterations on for
	const long *Rcol_ptr,
	const unsigned int *Rrow_idx,
	const float *Rval,
	float * u,
	float *v,
	const float lambda,
	float *innerfundec_cur,
	const int do_nmf,

	const long Rcols_t,
	const long *Rcol_ptr_t,
	const unsigned int *Rrow_idx_t,
	const float *Rval_t,
	float *innerfundec_cur2
	){

	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	float dev_innerfundec_cur = 0;
	float dev_innerfundec_cur2 = 0;
	innerfundec_cur[ii] = 0;
	innerfundec_cur2[ii] = 0;
	for (size_t c = ii; c < Rcols; c += blockDim.x*gridDim.x){
		v[c] = RankOneUpdate_dev(Rcol_ptr, Rrow_idx, Rval,
			c, u, lambda*(Rcol_ptr[c + 1] - Rcol_ptr[c]), v[c], &dev_innerfundec_cur, do_nmf);

	}
	innerfundec_cur[ii] = dev_innerfundec_cur;

	for (size_t c = ii; c < Rcols_t; c += blockDim.x*gridDim.x){
		u[c] = RankOneUpdate_dev(Rcol_ptr_t, Rrow_idx_t, Rval_t,
			c, v, lambda*(Rcol_ptr_t[c + 1] - Rcol_ptr_t[c]), u[c], &dev_innerfundec_cur2, do_nmf);

	}
	innerfundec_cur2[ii] = dev_innerfundec_cur2;

}

__device__ float RankOneUpdate_dev(const long *Rcol_ptr,
	const unsigned *Rrow_idx,
	const float *Rval,
	const int j,
	const float * u_vec_t,

	const float lambda,
	const float vj,
	float *redvar,
	const int do_nmf){

	float g = 0, h = lambda;
	if (Rcol_ptr[j + 1] == Rcol_ptr[j]) return 0;
	for (long idx = Rcol_ptr[j]; idx <Rcol_ptr[j + 1]; ++idx) {
		int i = Rrow_idx[idx];
		g += u_vec_t[i] * Rval[idx];
		h += u_vec_t[i] * u_vec_t[i];
	}
	float newvj = g / h, delta = 0, fundec = 0;
	if (do_nmf>0 & newvj < 0) {
		newvj = 0;
		delta = vj; // old - new
		fundec = -2 * g*vj; //+h*vj*vj;
	}
	else {
		delta = vj - newvj;
		fundec = h*delta*delta;
	}
	*redvar += fundec;
	return newvj;
}

__global__ void UpdateRating_DUAL_kernel_NoLoss(const long Rcols, //are the iterations on for
	const long *Rcol_ptr,
	const unsigned int *Rrow_idx,
	float *Rval,
	const float * Wt_vec_t,
	const float * Ht_vec_t,
	const bool add,

	const long Rcols_t, //are the iterations on for
	const long *Rcol_ptr_t,
	const unsigned int *Rrow_idx_t,
	float *Rval_t,
	const bool add_t
	){
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	//__threadfence_system();
	for (size_t i = ii; i < Rcols; i += blockDim.x*gridDim.x){
		if (add) {
			float Htc = Ht_vec_t[i];
			for (size_t idx = Rcol_ptr[i]; idx < Rcol_ptr[i + 1]; ++idx){
				Rval[idx] += Wt_vec_t[Rrow_idx[idx]] * Htc;//change R.val
			}
		}
		else {
			float Htc = Ht_vec_t[i];
			for (size_t idx = Rcol_ptr[i]; idx < Rcol_ptr[i + 1]; ++idx){
				Rval[idx] -= Wt_vec_t[Rrow_idx[idx]] * Htc;//change R.val
			}
		}
	}

	for (size_t i = ii; i < Rcols_t; i += blockDim.x*gridDim.x){
		if (add_t) {
			float Htc = Wt_vec_t[i];
			for (size_t idx = Rcol_ptr_t[i]; idx < Rcol_ptr_t[i + 1]; ++idx){
				Rval_t[idx] += Ht_vec_t[Rrow_idx_t[idx]] * Htc;//change R.val
			}
		}
		else {
			float Htc = Wt_vec_t[i];
			for (size_t idx = Rcol_ptr_t[i]; idx < Rcol_ptr_t[i + 1]; ++idx){
				Rval_t[idx] -= Ht_vec_t[Rrow_idx_t[idx]] * Htc;//change R.val
			}
		}
	}
}

void kernel_wrapper_ccdpp_NV(smat_t_C &R_C, float ** &W, float ** &H, params &parameters){
	cudaError_t cudaStatus = ccdpp_NV(R_C, W, H, parameters);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ALS FAILED: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceReset();
}

// Helper function for using CUDA.
cudaError_t ccdpp_NV(smat_t_C &R_C, float ** &W, float ** &H, params &parameters)
{
	long *dev_Rcol_ptr = 0;
	unsigned *dev_Rrow_idx = 0;
	long *dev_Rcol_ptr_T = 0;
	unsigned *dev_Rrow_idx_T = 0;
	float *dev_Rval = 0;
	float *dev_Rval_t = 0;
	float *dev_Wt_vec_t = 0;
	float *dev_Ht_vec_t = 0;

	float *dev_return = 0;
	float *dev_return2 = 0;
	float *Hostreduction = 0;
	float *Hostreduction2 = 0;


	int nThreadsPerBlock = parameters.nThreadsPerBlock;
	int nBlocks = parameters.nBlocks;
	cudaError_t cudaStatus;
	Hostreduction = (float*)malloc(sizeof(float)*(nThreadsPerBlock*nBlocks));
	Hostreduction2 = (float*)malloc(sizeof(float)*(nThreadsPerBlock*nBlocks));


	int k = parameters.k;
	int maxiter = parameters.maxiter;
	int inneriter = parameters.inneriter;
	float lambda = parameters.lambda;
	float eps = parameters.eps;
	long num_updates = 0;


	// Create transpose view of R
	smat_t_C Rt;
	Rt = transpose(R_C);
	// initial value of the regularization term
	// H is a zero matrix now.
	for (int t = 0; t<k; ++t) for (long c = 0; c<R_C.cols; ++c) H[t][c] = 0;

	float *u, *v;
	u = (float*)malloc(R_C.rows*sizeof(float));
	size_t nbits_u = R_C.rows*sizeof(float);
	v = (float*)malloc(R_C.cols*sizeof(float));
	size_t nbits_v = R_C.cols*sizeof(float);

	// Reset GPU.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Allocate GPU buffers for all vectors.
	cudaStatus = cudaMalloc((void**)&dev_Rcol_ptr, R_C.nbits_col_ptr);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Rrow_idx, R_C.nbits_row_idx);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Rcol_ptr_T, Rt.nbits_col_ptr);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Rrow_idx_T, Rt.nbits_row_idx);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Rval, R_C.nbits_val);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Rval_t, Rt.nbits_val);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Wt_vec_t, nbits_u);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Ht_vec_t, nbits_v);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_return, nThreadsPerBlock*nBlocks * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_return2, nThreadsPerBlock*nBlocks * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	// Copy all vectors to GPU buffers.
	cudaStatus = cudaMemcpy(dev_Rval, R_C.val, R_C.nbits_val, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Rval_t, Rt.val, Rt.nbits_val, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Rcol_ptr, R_C.col_ptr, R_C.nbits_col_ptr, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Rrow_idx, R_C.row_idx, R_C.nbits_row_idx, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Rcol_ptr_T, Rt.col_ptr, Rt.nbits_col_ptr, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Rrow_idx_T, Rt.row_idx, Rt.nbits_row_idx, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	for (int oiter = 1; oiter <= maxiter; ++oiter) {
		float rankfundec = 0;
		float fundec_max = 0;
		int early_stop = 0;
		for (int t = 0; t < k; ++t) {
			if (early_stop >= 5) break;
			float *Wt = &W[t][0], *Ht = &H[t][0];
			
			cudaStatus = cudaMemcpy(dev_Wt_vec_t, Wt, nbits_u, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			cudaStatus = cudaMemcpy(dev_Ht_vec_t, Ht, nbits_v, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			if (oiter > 1) {
				UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx, dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, true, Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, true);
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
					goto Error;
				}

			}

			float innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter;
			for (int iter = 1; iter <= maxit; ++iter){
				innerfundec_cur = 0;

				RankOneUpdate_DUAL_kernel<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx, dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, lambda, dev_return, parameters.do_nmf, Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, dev_return2);

				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
					goto Error;
				}

				cudaStatus = cudaMemcpy(Hostreduction, dev_return, nBlocks*nThreadsPerBlock * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!");
					goto Error;
				}

				cudaStatus = cudaMemcpy(Hostreduction2, dev_return2, nBlocks*nThreadsPerBlock * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!");
					goto Error;
				}

				for (size_t index = 0; index < nThreadsPerBlock*nBlocks; index++){
					innerfundec_cur += Hostreduction[index];
				}

				for (size_t index = 0; index < nThreadsPerBlock*nBlocks; index++){
					innerfundec_cur += Hostreduction2[index];
				}

				num_updates += R_C.cols;
				
				num_updates += Rt.cols;
				if ((innerfundec_cur < fundec_max*eps))  {
					if (iter == 1) early_stop += 1;
					break;
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = maxC(innerfundec_max, innerfundec_cur);
				if (!(oiter == 1 && t == 0 && iter == 1))
					fundec_max = maxC(fundec_max, innerfundec_cur);
			}

			cudaStatus = cudaMemcpy(Wt, dev_Wt_vec_t, nbits_u, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}

			cudaStatus = cudaMemcpy(Ht, dev_Ht_vec_t, nbits_v, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}


			UpdateRating_DUAL_kernel_NoLoss<<<nBlocks, nThreadsPerBlock>>>(R_C.cols, dev_Rcol_ptr, dev_Rrow_idx, dev_Rval, dev_Wt_vec_t, dev_Ht_vec_t, false, Rt.cols, dev_Rcol_ptr_T, dev_Rrow_idx_T, dev_Rval_t, false);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
				goto Error;
			}

		}
	}

Error:
	free(u);
	free(v);
	free(Hostreduction);
	free(Hostreduction2);

	cudaFree(dev_Rcol_ptr);
	cudaFree(dev_Rrow_idx);
	cudaFree(dev_Rcol_ptr_T);
	cudaFree(dev_Rrow_idx_T);
	cudaFree(dev_Rval);
	cudaFree(dev_Rval_t);
	cudaFree(dev_Wt_vec_t);
	cudaFree(dev_Ht_vec_t);
	cudaFree(dev_return);
	return cudaStatus;
}

smat_t_C transpose(smat_t_C m){
	smat_t_C mt;
	mt.cols = m.rows; mt.rows = m.cols; mt.nnz = m.nnz;
	mt.val = m.val_t; mt.val_t = m.val;
	mt.nbits_val = m.nbits_val_t; mt.nbits_val_t = m.nbits_val;
	mt.with_weights = m.with_weights;
	mt.weight = m.weight_t; mt.weight_t = m.weight;
	mt.nbits_weight = m.nbits_weight_t; mt.nbits_weight_t = m.nbits_weight;
	mt.col_ptr = m.row_ptr; mt.row_ptr = m.col_ptr;
	mt.nbits_col_ptr = m.nbits_row_ptr; mt.nbits_row_ptr = m.nbits_col_ptr;
	mt.col_idx = m.row_idx; mt.row_idx = m.col_idx;
	mt.nbits_col_idx = m.nbits_row_idx; mt.nbits_row_idx = m.nbits_col_idx;
	mt.max_col_nnz = m.max_row_nnz; mt.max_row_nnz = m.max_col_nnz;
	return mt;
}

float maxC(float a, float b){
	return(a>b ? a : b);
}
