#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "lif_kernel.h"


void fusedForwardLIF(
    const torch::Tensor& tX,  
    torch::Tensor& V_tV,
    torch::Tensor& tY,
    const int timeStep,
    const float decay, 
    const float threshold,
    const float rest,
    bool use_tV
) {
    const size_t tensorSize = tX[0].numel();
    launch_fusedForwardLIFKernel(
        tX.data_ptr<float>(),
        V_tV.data_ptr<float>(),
        tY.data_ptr<float>(),
        timeStep,
        tensorSize,
        decay,
        threshold,
        rest,
        use_tV
    );
}


void fusedBackwardLIF(
    const torch::Tensor& gtY,
    torch::Tensor& gtX,
    const torch::Tensor& tY,
    const torch::Tensor& tV,
    const int timeStep,
    const float decay, 
    const float threshold
) {
    const size_t tensorSize = tY[0].numel();
    launch_fusedBackwardLIFKernel(
        gtY.data_ptr<float>(),
        gtX.data_ptr<float>(),
        tY.data_ptr<float>(),
        tV.data_ptr<float>(),
        timeStep,
        tensorSize,
        decay,
        threshold
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fusedForwardLIF", &fusedForwardLIF, "forwardLIFKernel wrapper");
    m.def("fusedBackwardLIF", &fusedBackwardLIF, "backwardLIFKernel wrapper");
}
