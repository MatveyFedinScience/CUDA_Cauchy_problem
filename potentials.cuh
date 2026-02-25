#ifndef POTENTIALS_CUH
#define POTENTIALS_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ float potential(float x, float y) {
    return -0.173*x*x + -0.035*y*y + -0.021*x*y + -0.054*x;
}

__device__ __forceinline__ void gradient_potential(float x, float y, float* dPhi_dx, float* dPhi_dy) {
    *dPhi_dx = 2*-0.173*x + -0.021*y + -0.054;
    *dPhi_dy = 2*-0.035*y + -0.021*x;
}

#endif