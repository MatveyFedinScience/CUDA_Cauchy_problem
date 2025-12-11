#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "types.h"
#include "config.h"
#include "kernels.h"

int main() {

    Particle* d_particles;
    float* d_energies;

    cudaMalloc((void**)&d_particles, N_PARTICLES * sizeof(Particle));
    cudaMalloc((void**)&d_energies, N_PARTICLES * sizeof(float));

    int blocks = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float initial_velocity = 0.5f;
    int velocity_mode = 0;

    init_particles_kernel<<<blocks, BLOCK_SIZE>>>(
        d_particles, N_PARTICLES, R_CIRCLE, initial_velocity, velocity_mode
    );
    cudaDeviceSynchronize();

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
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Integration time: %.2f мс\n", milliseconds);

    compute_energy_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, d_energies, N_PARTICLES);
    cudaMemcpy(h_energies, d_energies, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);

    float E_final = 0.0f;
    for (int i = 0; i < N_PARTICLES; i++) {
        E_final += h_energies[i];
    }
    E_final /= N_PARTICLES;
    printf("Mean Finally Energy: %.6f\n", E_final);
    printf("|E_final - E_initial|/Energy: %.2e\n\n", fabsf(E_final - E_initial) / fabsf(E_initial));

    Particle* h_particles = (Particle*)malloc(N_PARTICLES * sizeof(Particle));

    cudaMemcpy(h_particles, d_particles, N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    free(h_particles);
    free(h_energies);

    cudaFree(d_particles);
    cudaFree(d_energies);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
