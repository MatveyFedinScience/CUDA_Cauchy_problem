#ifndef SOLVERS_CUH
#define SOLVERS_CUH

#include "potentials.cuh"

#include <cuda_runtime.h>

__device__ __forceinline__ void rk4_step(float* x, float* y, float* vx, float* vy, float dt) {
    float dPhi_dx, dPhi_dy;

    // k1
    float k1_x = *vx;
    float k1_y = *vy;
    gradient_potential(*x, *y, &dPhi_dx, &dPhi_dy);
    float k1_vx = -dPhi_dx;
    float k1_vy = -dPhi_dy;

    // k2
    float x2 = *x + 0.5f * dt * k1_x;
    float y2 = *y + 0.5f * dt * k1_y;
    float vx2 = *vx + 0.5f * dt * k1_vx;
    float vy2 = *vy + 0.5f * dt * k1_vy;

    float k2_x = vx2;
    float k2_y = vy2;
    gradient_potential(x2, y2, &dPhi_dx, &dPhi_dy);
    float k2_vx = -dPhi_dx;
    float k2_vy = -dPhi_dy;

    // k3
    float x3 = *x + 0.5f * dt * k2_x;
    float y3 = *y + 0.5f * dt * k2_y;
    float vx3 = *vx + 0.5f * dt * k2_vx;
    float vy3 = *vy + 0.5f * dt * k2_vy;

    float k3_x = vx3;
    float k3_y = vy3;
    gradient_potential(x3, y3, &dPhi_dx, &dPhi_dy);
    float k3_vx = -dPhi_dx;
    float k3_vy = -dPhi_dy;

    // k4
    float x4 = *x + dt * k3_x;
    float y4 = *y + dt * k3_y;
    float vx4 = *vx + dt * k3_vx;
    float vy4 = *vy + dt * k3_vy;

    float k4_x = vx4;
    float k4_y = vy4;
    gradient_potential(x4, y4, &dPhi_dx, &dPhi_dy);
    float k4_vx = -dPhi_dx;
    float k4_vy = -dPhi_dy;

    // step
    *x  += dt / 6.0f * (k1_x  + 2.0f*k2_x  + 2.0f*k3_x  + k4_x);
    *y  += dt / 6.0f * (k1_y  + 2.0f*k2_y  + 2.0f*k3_y  + k4_y);
    *vx += dt / 6.0f * (k1_vx + 2.0f*k2_vx + 2.0f*k3_vx + k4_vx);
    *vy += dt / 6.0f * (k1_vy + 2.0f*k2_vy + 2.0f*k3_vy + k4_vy);
}

#endif
