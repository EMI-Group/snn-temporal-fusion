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
);

void launch_fusedBackwardLIFKernel(
    const float* gtY, 
    float* gtX, 
    const float* tY, 
    const float* tV, 
    const int timeStep, 
    const size_t tensorSize, 
    const float decay, 
    const float threshold
);
