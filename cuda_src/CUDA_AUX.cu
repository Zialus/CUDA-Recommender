#include "CUDA_AUX.h"

__global__ void
GPU_rmse(long const* __restrict__ test_row, long const* __restrict__ test_col, float const* __restrict__ test_val,
         float* __restrict__ pred_v, float* __restrict__ rmse, float const* __restrict__ W,
         float const* __restrict__ H, int m, int k, int rows, int cols, bool ifALS) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m) {

        if (ifALS) {
            for (int t = 0; t < k; t++) {
                long i = test_row[c];
                long j = test_col[c];
                pred_v[c] += W[i * k + t] * H[j * k + t]; //W[i][t] * H[j][t]
//                pred_v[c] += W[t * rows + i] * H[t * cols + j]; //W[i][t] * H[j][t];
            }
        } else {
            for (int t = 0; t < k; t++) {
                long i = test_row[c];
                long j = test_col[c];
                pred_v[c] += W[t * rows + i] * H[t * cols + j]; //W[t][i] * H[t][j];
//                pred_v[c] += W[i * t + rows] * H[j * t + cols]; //W[t][i] * H[t][j];
            }
        }
        rmse[c] = (pred_v[c] - test_val[c]) * (pred_v[c] - test_val[c]);
    }
}
