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


// Gauss-Legendre constants
#define GL_C1   0.2113248654051871f  // 0.5 - sqrt(3)/6
#define GL_C2   0.7886751345948129f  // 0.5 + sqrt(3)/6
#define GL_A11  0.25f
#define GL_A12 -0.0386751345948129f  // 0.25 - sqrt(3)/6
#define GL_A21  0.5386751345948129f  // 0.25 + sqrt(3)/6
#define GL_A22  0.25f

__device__ __forceinline__ void gauss_legendre_step(
    float* x, float* y, float* vx, float* vy, float dt,
    int max_iter = 8, float tol = 1e-6f
) {
    float x0 = *x, y0 = *y, vx0 = *vx, vy0 = *vy;

    float dPhi_dx, dPhi_dy;
    gradient_potential(x0, y0, &dPhi_dx, &dPhi_dy);

    // k1 = (k1_x, k1_y, k1_vx, k1_vy)
    // k2 = (k2_x, k2_y, k2_vx, k2_vy)
    float k1_x = vx0, k1_y = vy0, k1_vx = -dPhi_dx, k1_vy = -dPhi_dy;
    float k2_x = vx0, k2_y = vy0, k2_vx = -dPhi_dx, k2_vy = -dPhi_dy;

    for (int iter = 0; iter < max_iter; iter++) {

        float k1_x_old = k1_x, k1_y_old = k1_y;
        float k2_x_old = k2_x, k2_y_old = k2_y;

        //Y1 = Y0 + dt*(a11*k1 + a12*k2)
        float x1  = x0  + dt * (GL_A11 * k1_x  + GL_A12 * k2_x);
        float y1  = y0  + dt * (GL_A11 * k1_y  + GL_A12 * k2_y);
        float vx1 = vx0 + dt * (GL_A11 * k1_vx + GL_A12 * k2_vx);
        float vy1 = vy0 + dt * (GL_A11 * k1_vy + GL_A12 * k2_vy);

        // k1 = f(Y1)
        k1_x = vx1;
        k1_y = vy1;
        gradient_potential(x1, y1, &dPhi_dx, &dPhi_dy);
        k1_vx = -dPhi_dx;
        k1_vy = -dPhi_dy;

        //Y2 = Y0 + dt*(a21*k1 + a22*k2)
        float x2  = x0  + dt * (GL_A21 * k1_x  + GL_A22 * k2_x);
        float y2  = y0  + dt * (GL_A21 * k1_y  + GL_A22 * k2_y);
        float vx2 = vx0 + dt * (GL_A21 * k1_vx + GL_A22 * k2_vx);
        float vy2 = vy0 + dt * (GL_A21 * k1_vy + GL_A22 * k2_vy);

        // k2 = f(Y2)
        k2_x = vx2;
        k2_y = vy2;
        gradient_potential(x2, y2, &dPhi_dx, &dPhi_dy);
        k2_vx = -dPhi_dx;
        k2_vy = -dPhi_dy;

        float err = fabsf(k1_x - k1_x_old) + fabsf(k1_y - k1_y_old) +
                    fabsf(k2_x - k2_x_old) + fabsf(k2_y - k2_y_old);
        if (err < tol) break;
    }

    *x  = x0  + 0.5f * dt * (k1_x  + k2_x);
    *y  = y0  + 0.5f * dt * (k1_y  + k2_y);
    *vx = vx0 + 0.5f * dt * (k1_vx + k2_vx);
    *vy = vy0 + 0.5f * dt * (k1_vy + k2_vy);
}

#endif
