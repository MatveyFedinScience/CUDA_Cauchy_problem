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
#include "helpers.cuh"


int main() {
    print_device_info();

    printf("=== Simulation Parameters ===\n");
    printf("Particles: %d\n", N_PARTICLES);
    printf("Time step: %e\n", DT);
    printf("Total steps: %d\n", N_STEPS);
    printf("Total simulation time: %f\n", (float)DT * N_STEPS);
//    printf("Note: Due to optimization, grad(potential) != force on boundary.\n");
//    printf("      This causes initial/final energy inequality.\n");
    printf("=============================\n\n");

    Particle* d_particles;
    curandState* d_states;
    float* d_energies;

    CUDA_CHECK(cudaMalloc(&d_particles, N_PARTICLES * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_states, N_PARTICLES * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_energies, N_PARTICLES * sizeof(float)));

    Particle* h_particles = (Particle*)malloc(N_PARTICLES * sizeof(Particle));
    float* h_energies = (float*)malloc(N_PARTICLES * sizeof(float));
    char* image = NULL;

    if (!h_particles || !h_energies) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    const int blocks = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const unsigned long long seed = 1;

    const int side = (int)ceilf(sqrtf((float)N_PARTICLES));
    dim3 init_block(16, 16);
    dim3 init_grid(
        (side + init_block.x - 1) / init_block.x,
        (side + init_block.y - 1) / init_block.y
    );
    printf("Init grid: [%d, %d], block: [%d, %d]\n",
           init_grid.x, init_grid.y, init_block.x, init_block.y);

    setup_curand_states_kernel<<<blocks, BLOCK_SIZE>>>(d_states, seed, N_PARTICLES);
    CUDA_CHECK(cudaGetLastError());

    init_particles_kernel<<<init_grid, init_block>>>(
        d_particles, d_states, N_PARTICLES, R_CIRCLE, MAX_INIT_VEL, MIN_INIT_VEL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_particles, d_particles,
                          N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));
    init_particles_to_image(h_particles, N_PARTICLES, &image);


    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);
    CUDA_CHECK(cudaGetLastError());

    float E_initial = compute_mean_energy(d_energies, h_energies, N_PARTICLES);
    printf("\nMean initial energy: %.6f\n", E_initial);


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    integrate_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N_PARTICLES, DT, N_STEPS);
    CUDA_CHECK(cudaGetLastError());

    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);
    CUDA_CHECK(cudaGetLastError());

    finalize_particles_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N_PARTICLES);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Integration time: %.2f ms\n", milliseconds);

    float E_before_finalize = compute_mean_energy(d_energies, h_energies, N_PARTICLES);
    printf("Mean energy (before finalization): %.6f\n", E_before_finalize);

    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);
    CUDA_CHECK(cudaGetLastError());

    float E_final = compute_mean_energy(d_energies, h_energies, N_PARTICLES);
    printf("Mean final energy: %.6f\n", E_final);

    float relative_error = fabsf(E_final - E_initial) / fabsf(E_initial);
    printf("Relative energy error: %.2e\n\n", relative_error);

    CUDA_CHECK(cudaMemcpy(h_particles, d_particles,
                          N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));

    long long total_steps = 0;
    for (int i = 0; i < N_PARTICLES; i++) {
        total_steps += h_particles[i].steps;
    }
    float mean_sim_time = ((float)total_steps / N_PARTICLES) * DT;
    printf("Mean simulation time per particle: %f\n", mean_sim_time);

    final_particles_to_image(h_particles, N_PARTICLES, &image);
    save_single_image("test.ppm", image, 256, 256);

    free(h_particles);
    free(h_energies);
    free(image);

    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_energies));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
