#ifndef POTENTIALS_CUH
#define POTENTIALS_CUH

#include <cuda_runtime.h>
#include "FastNoiseLiteCUDA.h"
#include "config.h"

extern __device__ cudaTextureObject_t d_globalNoiseTex;
extern __device__ cudaTextureObject_t d_FxTex;
extern __device__ cudaTextureObject_t d_FyTex;
extern __device__ bool g_use_forces;

__device__ __forceinline__ float clampf(float val, float lo, float hi) {
    return fminf(fmaxf(val, lo), hi);
}

__device__ __forceinline__ float potential(float X, float Y) {
    if (g_use_forces) return 0.0f;
    float u = (X + 1.0f) * 0.5f * NOISE_WIDTH;
    float v = (Y + 1.0f) * 0.5f * NOISE_HEIGHT;

    float val = tex2D<float>(d_globalNoiseTex, u + 0.5f, v + 0.5f) * IN_CIRCLE(X, Y);
    return clampf(val, -POTENTIAL_CLAMP, POTENTIAL_CLAMP);
}

__device__ __forceinline__ void gradient_potential(float X, float Y, float* dPhi_dx, float* dPhi_dy) {
    if (g_use_forces) {
        float u = (X + 1.0f) * 0.5f * NOISE_WIDTH;
        float v = (Y + 1.0f) * 0.5f * NOISE_HEIGHT;
        float fx = -tex2D<float>(d_FxTex, u + 0.5f, v + 0.5f) * IN_CIRCLE(X, Y);
        float fy = -tex2D<float>(d_FyTex, u + 0.5f, v + 0.5f) * IN_CIRCLE(X, Y);
        *dPhi_dx = clampf(fx, -GRADIENT_CLAMP, GRADIENT_CLAMP);
        *dPhi_dy = clampf(fy, -GRADIENT_CLAMP, GRADIENT_CLAMP);
        return;
    }
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

    float gx = grad_u * scale * IN_CIRCLE(X, Y);
    float gy = grad_v * scale * IN_CIRCLE(X, Y);

    *dPhi_dx = clampf(gx, -GRADIENT_CLAMP, GRADIENT_CLAMP);
    *dPhi_dy = clampf(gy, -GRADIENT_CLAMP, GRADIENT_CLAMP);
}

#if BLUR_ENABLED

static __device__ float gaussian_2d_weight(int dx, int dy, float sigma) {
    float r2 = (float)(dx * dx + dy * dy);
    return __expf(-r2 / (2.0f * sigma * sigma));
}

static __global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    const int R = BLUR_RADIUS;
    const float sigma = BLUR_SIGMA;

    for (int dy = -R; dy <= R; dy++) {
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= height) sy = height - 1;

        for (int dx = -R; dx <= R; dx++) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= width) sx = width - 1;

            float w = gaussian_2d_weight(dx, dy, sigma);
            sum += input[sy * width + sx] * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = sum / weight_sum;
}

#endif

#endif
