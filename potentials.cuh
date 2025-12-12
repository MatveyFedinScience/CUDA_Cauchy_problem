#ifndef POTENTIALS_CUH
#define POTENTIALS_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ float potential(float x, float y) {
    return 0*0.5f * (x * x + y * y) * IN_CIRCLE(x, y);
}

__device__ __forceinline__ void gradient_potential(float x, float y, float* dPhi_dx, float* dPhi_dy) {
    *dPhi_dx = 0*x * IN_CIRCLE(x, y);
    *dPhi_dy = 0*y * IN_CIRCLE(x, y);
}

//#define POTENTIAL(x,y) (0.5f * ((x) * (x) + (y) * (y)))

#endif
