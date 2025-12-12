#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "types.h"
#include "config.h"
#include "kernels.h"

int main() {

    printf("Due to optimisation reasons \\nabla potential != force on the boundary. It's reason of unequity of initial and final energy\n");
    Particle* d_particles;
    float* d_energies;

    cudaMalloc((void**)&d_particles, N_PARTICLES * sizeof(Particle));
    cudaMalloc((void**)&d_energies, N_PARTICLES * sizeof(float));

    int blocks = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float initial_velocity = 0.5f;
    float phi = 1 * M_PI / 2;

    init_particles_kernel<<<blocks, BLOCK_SIZE>>>(
        d_particles, N_PARTICLES, R_CIRCLE, initial_velocity, phi
    );

    Particle* h_particles = (Particle*)malloc(N_PARTICLES * sizeof(Particle));

    cudaMemcpy(h_particles, d_particles, N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

/*    float x;
    float y;

    for (int i = 0; i < 100; i++){

        x = h_particles[i].x;
        y = h_particles[i].y;
        x += y;
//        printf("%.6f\n", x);
  //      printf("%.6f\n", y);
    //    printf("%.6f\n", x * x + y * y);

    }
*/
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

//    Particle* h_particles = (Particle*)malloc(N_PARTICLES * sizeof(Particle));

    cudaMemcpy(h_particles, d_particles, N_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    int mean_time = 0;

    for (int i = 0; i < N_PARTICLES; i++){

        mean_time += h_particles[i].steps;

    }

    printf("%f\n",(float) (mean_time / N_PARTICLES) * DT);
    free(h_particles);
    free(h_energies);

    cudaFree(d_particles);
    cudaFree(d_energies);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
