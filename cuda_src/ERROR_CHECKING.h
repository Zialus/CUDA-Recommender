#ifndef ERROR_CHECKING_H
#define ERROR_CHECKING_H

#include <cassert>
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s - %s %d\n", cudaGetErrorString(code), file, line);
        assert(code == cudaSuccess);
    }
}

__global__ void
GPU_rmse(long const* __restrict__ test_row, long const* __restrict__ test_col, float const* __restrict__ test_val,
         float* __restrict__ pred_v, float* __restrict__ rmse, float const* __restrict__ W,
         float const* __restrict__ H, int m, int k, int rows, int cols, bool ifALS);

#endif //ERROR_CHECKING_H
