#ifndef POTENTIALS_CUH
#define POTENTIALS_CUH

#include <cuda_runtime.h>
#include "FastNoiseLiteCUDA.h"
#include "config.h"

extern __device__ cudaTextureObject_t d_globalNoiseTex;


__device__ __forceinline__ float potential(float X, float Y) {
    float u = (X + 1.0f) * 0.5f * NOISE_WIDTH;
    float v = (Y + 1.0f) * 0.5f * NOISE_HEIGHT;

    return tex2D<float>(d_globalNoiseTex, u + 0.5f, v + 0.5f) * IN_CIRCLE(X, Y);
}

__device__ __forceinline__ void gradient_potential(float X, float Y, float* dPhi_dx, float* dPhi_dy) {
    float u = (X + 1.0f) * 0.5f * NOISE_WIDTH;
    float v = (Y + 1.0f) * 0.5f * NOISE_HEIGHT;
  
    const float eps = 1.0f; 
 
    float val_left   = tex2D<float>(d_globalNoiseTex, u - eps + 0.5f, v + 0.5f);
    float val_right  = tex2D<float>(d_globalNoiseTex, u + eps + 0.5f, v + 0.5f);
    float val_top    = tex2D<float>(d_globalNoiseTex, u + 0.5f, v - eps + 0.5f);
    float val_bottom = tex2D<float>(d_globalNoiseTex, u + 0.5f, v + eps + 0.5f);

    float grad_u = (val_right - val_left) * 0.5f; 
    float grad_v = (val_bottom - val_top) * 0.5f;

    float scale = NOISE_SCALE * NOISE_WIDTH * 0.5f; 

    *dPhi_dx = grad_u * scale * IN_CIRCLE(X, Y);
    *dPhi_dy = grad_v * scale * IN_CIRCLE(X, Y);
}


/*
__device__ __forceinline__ float potential(float x, float y) {
    return -0.173*x*x + -0.035*y*y + -0.021*x*y + -0.054*x;
}

__device__ __forceinline__ void gradient_potential(float x, float y, float* dPhi_dx, float* dPhi_dy) {
    *dPhi_dx = 2*-0.173*x + -0.021*y + -0.054;
    *dPhi_dy = 2*-0.035*y + -0.021*x;
}
*/
#endif
