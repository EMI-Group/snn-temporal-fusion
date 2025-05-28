#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "lif_kernel.h"


__global__ void fusedForwardLIFKernel_V(
    const float* tX, 
    float* V,        
    float* tY,       
    const int timeStep, 
    const size_t tensorSize, 
    const float decay, 
    const float threshold,
    const float rest
) { 
    const size_t tensorIndex = (size_t )blockIdx.x * blockDim.x + (size_t )threadIdx.x;
    if (tensorIndex >= tensorSize) return;

    float v = float{ rest };
    float y = (v >= threshold)? float{ 1 }: float{ 0 };
    for (int t = 0; t < timeStep; t++) {
        const size_t pos = t * tensorSize + tensorIndex;
        v = decay * v * (1 - y) + rest * y + tX[pos];
        y = (v >= threshold)? float{ 1 }: float{ 0 };
        tY[pos] = y;
    }
    V[tensorIndex] = v;
}


__global__ void fusedForwardLIFKernel_tV(
    const float* tX,
    float* tV,
    float* tY,
    const int timeStep, 
    const size_t tensorSize, 
    const float decay, 
    const float threshold,
    const float rest
) { 
    const size_t tensorIndex = (size_t )blockIdx.x * blockDim.x + (size_t )threadIdx.x;
    if (tensorIndex >= tensorSize) return;

    float v = rest;
    float y = (v >= threshold)? float{ 1 }: float{ 0 };
    for (int t = 0; t < timeStep; t++) {
        const size_t pos = t * tensorSize + tensorIndex;
        v = decay * v * (1 - y) + rest * y + tX[pos];
        y = (v >= threshold)? float{ 1 }: float{ 0 };
        tV[pos] = v;
        tY[pos] = y;
    }
}


void launch_fusedForwardLIFKernel(
    const float* tX, 
    float* V, 
    float* tY, 
    const int timeStep, 
    const size_t tensorSize, 
    const float decay, 
    const float threshold,
    const float rest,
    bool use_tV
) {
    cudaError_t err;
    int gridSize{}, blockSize{};
    auto lif_kernel = (use_tV)? fusedForwardLIFKernel_tV: fusedForwardLIFKernel_V;

    err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, lif_kernel);
    if (err != cudaSuccess) { 
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    if (gridSize * blockSize < tensorSize) {
        gridSize = (tensorSize - 1) / blockSize + 1;
    }
    lif_kernel<<<gridSize, blockSize>>>(tX, V, tY, timeStep, tensorSize, decay, threshold, rest);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
} 


__device__ float sigmoidSurrogate(
    const float scalarInput,
    const float threshold
) {
    float alpha = 4.0;
    return alpha / 2 / (1 + coshf(alpha * scalarInput));
}

__device__ float hardSigmoidSurrogate(
    const float scalarInput,
    const float threshold,
    const float lens
) {
    return fabs(scalarInput-threshold) < lens? float{ 1 }: float{ 0 };
}


__global__ void fusedBackwardLIFKernel(
    const float* gtY,
    float* gtX,
    const float* tY,
    const float* tV,
    const int timeStep, 
    const size_t tensorSize, 
    const float decay, 
    const float threshold
) {
    const size_t tensorIndex = (size_t )blockIdx.x * blockDim.x + (size_t )threadIdx.x;
    if (tensorIndex >= tensorSize) return;

    float l2v = 0;
    const float lens = 0.5;
    for (int t=timeStep - 1; t >= 0; t--) {
        const size_t pos = t * tensorSize + tensorIndex;
        const float y2v = hardSigmoidSurrogate(tV[pos], threshold, lens); 
        const float l2y = gtY[pos];
        const float v2v = decay*(1 - tY[pos] - tV[pos] * y2v);
        l2v = l2y * y2v + l2v * v2v;
        gtX[pos] = l2v;
    }
}

void launch_fusedBackwardLIFKernel(
    const float* gtY,
    float* gtX,
    const float* tY,
    const float* tV,
    const int timeStep, 
    const size_t tensorSize, 
    const float decay, 
    const float threshold
) {
    cudaError_t err;
    int gridSize{}, blockSize{};
    err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedBackwardLIFKernel);
    if (err != cudaSuccess) { 
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    if (gridSize*blockSize < tensorSize) {
        gridSize = (tensorSize - 1) / blockSize + 1;
    }
    fusedBackwardLIFKernel<<<gridSize, blockSize>>>(gtY, gtX, tY, tV, timeStep, tensorSize, decay, threshold);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
}
