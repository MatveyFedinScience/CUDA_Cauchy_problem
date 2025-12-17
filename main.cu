#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


#include "types.h"
#include "config.h"
#include "kernels.h"
#include "helpers.h"

int main() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);

    printf("maxThreadsPerBlock = %d\n", p.maxThreadsPerBlock);
    printf("maxThreadsDim      = %d %d %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("maxGridSize        = %d %d %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

    printf("Due to optimisation reasons \\nabla potential != force on the boundary. It's reason of unequity of initial and final energy\n");
    printf("Total simulation time is %f\n", (float) DT * N_STEPS);
    Particle* d_particles;
    curandState* d_states;
    float* d_energies;

    cudaMalloc((void**)&d_particles, N_PARTICLES * sizeof(Particle));
    cudaMalloc((void**)&d_states, N_PARTICLES * sizeof(curandState));
    cudaMalloc((void**)&d_energies, N_PARTICLES * sizeof(float));

    int blocks = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float max_init_vel = MAX_INIT_VEL;
    float min_init_vel = MIN_INIT_VEL;

    unsigned long long seed = 1; //time(NULL);
    setup_curand_states_kernel<<<blocks, BLOCK_SIZE>>>(d_states, seed, N_PARTICLES);

    dim3 block(16, 16);
    dim3 grid(
        (int)ceilf(sqrtf((float)N_PARTICLES) / block.x),
        (int)ceilf(sqrtf((float)N_PARTICLES) / block.y)
    );

    printf("%d\n", (int)ceilf(sqrtf((float)N_PARTICLES)));

    init_particles_kernel<<<grid, block>>>(
        d_particles, d_states, N_PARTICLES, R_CIRCLE, max_init_vel, min_init_vel);

    Particle* h_particles = (Particle*)malloc(N_PARTICLES * sizeof(Particle));

    cudaMemcpy(h_particles, d_particles, N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    char* image;
    init_particles_to_image(h_particles, N_PARTICLES, &image);

    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);

    float* h_energies = (float*)malloc(N_PARTICLES * sizeof(float));

    cudaMemcpy(h_energies, d_energies, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);

    float E_initial = 0.0f;
    for (int i = 0; i < N_PARTICLES; i++) {
        E_initial += h_energies[i];
    }
    E_initial /= N_PARTICLES;
    printf("Mean Init Energy: %.6f\n", E_initial);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    integrate_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N_PARTICLES, DT, N_STEPS);
    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);
    finalize_particles_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N_PARTICLES);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Integration time: %.2f ms\n", milliseconds);
    cudaMemcpy(h_energies, d_energies, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);
    float E_final = 0.0f;
    for (int i = 0; i < N_PARTICLES; i++) {
        E_final += h_energies[i];
    }
    E_final /= N_PARTICLES;
    printf("Mean Finally Energy before Finalization: %.6f\n", E_final);


    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);
    cudaMemcpy(h_energies, d_energies, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);

    E_final = 0.0f;
    for (int i = 0; i < N_PARTICLES; i++) {
        E_final += h_energies[i];
    }
    E_final /= N_PARTICLES;
    printf("Mean Finally Energy: %.6f\n", E_final);
    printf("|E_final - E_initial|/Energy: %.2e\n\n", fabsf(E_final - E_initial) / fabsf(E_initial));

    cudaMemcpy(h_particles, d_particles, N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    int mean_time = 0;

    for (int i = 0; i < N_PARTICLES; i++){

        mean_time += h_particles[i].steps;

    }


    printf("%f\n",(float) (mean_time / N_PARTICLES) * DT);
    final_particles_to_image(h_particles, N_PARTICLES, &image);
    save_single_image("test.ppm", image, 256, 256);
    free(h_particles);
    free(h_energies);

    cudaFree(d_particles);
    cudaFree(d_energies);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
