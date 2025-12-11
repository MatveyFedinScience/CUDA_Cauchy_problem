#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include "config.h"


__global__ void init_particles_kernel(Particle* particles, int n, float R,
                                       float v0, int velocity_mode);

__global__ void integrate_kernel(Particle* particles, int n, float dt, int n_steps);

__global__ void integrate_with_history_kernel(Particle* particles,
                                               float* trajectory_x,
                                               float* trajectory_y,
                                               int n, float dt,
                                               int n_steps, int save_every);

__global__ void compute_energy_kernel(Particle* particles, float* energies, int n);

#endif
