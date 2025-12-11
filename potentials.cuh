#ifndef POTENTIALS_CUH
#define POTENTIALS_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ float potential(float x, float y) {
    return 0.5f * (x * x + y * y);
}

__device__ __forceinline__ void gradient_potential(float x, float y, float* dPhi_dx, float* dPhi_dy) {
    *dPhi_dx = x;
    *dPhi_dy = y;
}

#endif
