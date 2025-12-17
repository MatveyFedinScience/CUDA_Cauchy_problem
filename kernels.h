#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include "config.h"


__global__ void setup_curand_states_kernel(curandState* states,
                                          unsigned long long seed, int n);

__global__ void init_particles_kernel(Particle* particles,
                                      curandState* states,
                                      int n,
                                      float R,
                                      float v_max,
                                      float v_min);

__global__ void integrate_kernel(Particle* particles, int n, float dt, int n_steps);

__global__ void integrate_with_history_kernel(Particle* particles,
                                               float* trajectory_x,
                                               float* trajectory_y,
                                               int n, float dt,
                                               int n_steps, int save_every);

__global__ void compute_energy_kernel(Particle* particles, float* energies, int n);

__global__ void finalize_particles_kernel(Particle* particles, int n);

#endif
