#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "kernels.h"
#include "calc_methods.cuh"

#include <math.h>

/**
* @brief Initializes an array of particles placed in a circle,
* and sets their initial velocities according to the selected mode.
*
* @param particles Pointer to an array of particles in the GPU global memory.
* @param n Number of particles.
* @param R Radius of the circle along which the particles will be placed.
* @param v0(temp) Base value of the initial velocity.
* @param phi Start angle for all particles relatively tangent in start point.
* @details
* Each particle is assigned a coordinate based on the angle
* θ = 2π * idx / n
* which distributes the particles evenly around the circumference.
*/
__global__ void init_particles_kernel(Particle* particles, int n, float R,
                                       float v0, float phi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if ((0 >= phi) && (phi >= M_PI)) printf("\nphi must be between zero and pi! But your phi is %f!\n", phi);

    float theta = 2.0f * M_PI * idx / n;

    particles[idx].x = R * cosf(theta);
    particles[idx].y = R * sinf(theta);
    particles[idx].steps = 0;

    particles[idx].vx = v0 * cosf(theta + 0.5f * M_PI + phi);
    particles[idx].vy = v0 * sinf(theta + 0.5f * M_PI + phi);
}

__global__ void integrate_kernel(Particle* particles, int n, float dt, int n_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = particles[idx].x;
    float y = particles[idx].y;
    float vx = particles[idx].vx;
    float vy = particles[idx].vy;

    for (int step = 0; step < n_steps; step++) {
        CALC_METHOD(&x, &y, &vx, &vy, dt);
        particles[idx].steps += IN_CIRCLE(x, y);
    }

    particles[idx].x = x;
    particles[idx].y = y;
    particles[idx].vx = vx;
    particles[idx].vy = vy;

}

__global__ void integrate_with_history_kernel(Particle* particles,
                                               float* trajectory_x,
                                               float* trajectory_y,
                                               int n, float dt,
                                               int n_steps, int save_every) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x  = particles[idx].x;
    float y  = particles[idx].y;
    float vx = particles[idx].vx;
    float vy = particles[idx].vy;

    int n_saved = (n_steps / save_every) + 1;

    trajectory_x[idx * n_saved] = x;
    trajectory_y[idx * n_saved] = y;

    int save_idx = 1;

    for (int step = 1; step <= n_steps; step++) {
        CALC_METHOD(&x, &y, &vx, &vy, dt);
        particles[idx].steps += IN_CIRCLE(x, y);
        if (step % save_every == 0 && save_idx < n_saved) {
            trajectory_x[idx * n_saved + save_idx] = x;
            trajectory_y[idx * n_saved + save_idx] = y;
            save_idx++;
        }
    }

    particles[idx].x  = x;
    particles[idx].y  = y;
    particles[idx].vx = vx;
    particles[idx].vy = vy;

}

__global__ void compute_energy_kernel(Particle* particles, float* energies, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x  = particles[idx].x;
    float y  = particles[idx].y;
    float vx = particles[idx].vx;
    float vy = particles[idx].vy;

    float kinetic = 0.5f * (vx * vx + vy * vy);
    float pot = potential(x*.99, y*.99);

    energies[idx] = kinetic + pot;

}

__global__ void finalize_particles_kernel(Particle* particles, int n) {
    //After this function you take final position and velocities of the particles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float vx  = particles[idx].vx;
    float vy  = particles[idx].vy;
    float tau = particles[idx].steps * DT;

    particles[idx].x -= ( N_STEPS * DT - tau ) * vx;
    particles[idx].y -= ( N_STEPS * DT - tau ) * vy;

}
