#ifndef CUDA_AUX_H
#define CUDA_AUX_H

#include <cassert>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s - %s %d\n", cudaGetErrorString(code), file, line);
        assert(code == cudaSuccess);
    }
}

__global__ void
GPU_rmse(unsigned const* __restrict__ test_row, unsigned const* __restrict__ test_col, float const* __restrict__ test_val,
         float* __restrict__ pred_v, float* __restrict__ rmse, float const* __restrict__ W,
         float const* __restrict__ H, int m, int k, int rows, int cols, bool ifALS);


struct GpuTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, nullptr);
    }

    void Stop() {
        cudaEventRecord(stop, nullptr);
    }


    float Elapsed() {
        float miliseconds;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&miliseconds, start, stop);
        return miliseconds / 1000;
    }

};

#endif //CUDA_AUX_H
