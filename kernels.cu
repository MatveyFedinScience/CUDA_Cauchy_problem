#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "kernels.h"
#include "calc_methods.cuh"

#include <math.h>



__global__ void setup_curand_states_kernel(curandState* states,
                                          unsigned long long seed,
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curand_init(seed, idx, 0, &states[idx]);
}



__global__ void init_random_particles_kernel(Particle* particles,
                                      curandState* states,
                                      int n,
                                      float R,
                                      float v0,
                                      float phi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (phi < 0.0f || phi > M_PI) {
        printf("\nphi must be between zero and pi! But your phi is %f!\n", phi);
        return;
    }

    curandState localState = states[idx];

    float theta = 2.0f * M_PI * curand_uniform(&localState);

    float cos_theta, sin_theta;
    __sincosf(theta, &sin_theta, &cos_theta);

    particles[idx].x = R * cos_theta;
    particles[idx].y = R * sin_theta;
    particles[idx].steps = 0;

    float angle_for_velocity = theta + (0.5f * M_PI + phi);

    float cos_vel, sin_vel;
    __sincosf(angle_for_velocity, &sin_vel, &cos_vel);

    particles[idx].vx = v0 * cos_vel;
    particles[idx].vy = v0 * sin_vel;
}


/**
* @brief Initializes an array of particles placed in a circle,
* and sets their initial velocities according to the selected mode.
*
* @param particles Pointer to an array of particles in the GPU global memory.
* @param n Number of particles.
* @param R Radius of the circle along which the particles will be placed.
* @param v_max value of the initial velocity.
* @details
* Each particle is assigned a coordinate based on the angle
* θ = 2π * idx / n
* which distributes the particles evenly around the circumference.
*/
__global__ void init_particles_kernel(Particle* particles,
                                      curandState* states,
                                      int n,
                                      float R,
                                      float v_max,
                                      float v_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int particle_idx = idx + idy * gridDim.x * blockDim.x;

    if (particle_idx >= n) return;

    float theta = 2.0f * M_PI * idx / (gridDim.x * blockDim.x);
    float phi   = M_PI * idy / (gridDim.y * blockDim.y);

    particles[particle_idx].x = R * cosf(theta);
    particles[particle_idx].y = R * sinf(theta);
    particles[particle_idx].steps = 0;

    curandState localState = states[particle_idx];

    float random_speed_factor = curand_uniform(&localState);  // [0.0, 1.0]
    float actual_v0 = v_min + ( v_max - v_min ) * random_speed_factor;

    states[particle_idx] = localState;

    float velocity_angle = theta + 0.5f * M_PI + phi;
    particles[particle_idx].vx = actual_v0 * cosf(velocity_angle);
    particles[particle_idx].vy = actual_v0 * sinf(velocity_angle);
}




__global__ void integrate_kernel(Particle* particles, int n, float dt, int n_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = particles[idx].x;
    float y = particles[idx].y;
    float vx = particles[idx].vx;
    float vy = particles[idx].vy;
    int local_steps = 0;

    for (int step = 0; step < n_steps; step++) {
        CALC_METHOD(&x, &y, &vx, &vy, dt);
        local_steps += IN_CIRCLE(x, y);
    }

    particles[idx].x = x;
    particles[idx].y = y;
    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].steps = local_steps;

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
    float vx = particles[idx].vx;// * IN_CIRCLE(x, y);
    float vy = particles[idx].vy;// * IN_CIRCLE(x, y);
    int local_steps = 0;

    int n_saved = (n_steps / save_every) + 1;

    trajectory_x[idx * n_saved] = x;
    trajectory_y[idx * n_saved] = y;

    int save_idx = 1;

    for (int step = 1; step <= n_steps; step++) {
        CALC_METHOD(&x, &y, &vx, &vy, dt);
        local_steps += IN_CIRCLE(x, y);
        if (step % save_every == 0 && save_idx < n_saved) {
            trajectory_x[idx * n_saved + save_idx] = x;
            trajectory_y[idx * n_saved + save_idx] = y;
            save_idx++;
        }
    }

    particles[idx].x  = x;
    particles[idx].y  = y;
    particles[idx].vx = vx;// + (1 - IN_CIRCLE(x, y)) * particles[idx].vx;
    particles[idx].vy = vy;// + (1 - IN_CIRCLE(x, y)) * particles[idx].vy;
    particles[idx].steps = local_steps;
}

__global__ void compute_energy_kernel(Particle* particles, float* energies, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x  = particles[idx].x;
    float y  = particles[idx].y;
    float vx = particles[idx].vx;
    float vy = particles[idx].vy;

    float kinetic = 0.5f * (vx * vx + vy * vy);
    float pot = potential(x, y);

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
