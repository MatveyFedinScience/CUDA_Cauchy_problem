#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#include "types.h"
#include "config.h"
#include "kernels.h"
#include "helpers.h"
#include "helpers.cuh"

#include "potentials.cuh"


__device__ cudaTextureObject_t d_globalNoiseTex; 



__global__ void generate_noise_kernel(float* output, int width, int height, int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float X = ((float)x / (float)width) * 2.0f - 1.0f;
    float Y = ((float)y / (float)height) * 2.0f - 1.0f;


    if (x >= width || y >= height)
    {
        return;
    }

    FastNoiseLite noise(seed); 
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFrequency(NOISE_FREQ);

    float noiseValue = NOISE_SCALE * ( noise.GetNoise((float)x, (float)y) ) * IN_CIRCLE(X, Y) - NOISE_SCALE * ( 1 - IN_CIRCLE(X, Y) );

    output[y * width + x] = noiseValue;
}


__global__ void debug_texture_check() {
    float val_center = tex2D<float>(d_globalNoiseTex, 512.5f, 512.5f);
    
    float val_offset = tex2D<float>(d_globalNoiseTex, 513.5f, 512.5f);
    
    unsigned long long texID = (unsigned long long)d_globalNoiseTex;

    printf("\n=== GPU DEBUG INFO ===\n");
    printf("Texture Object ID: %llu (if 0 - texture creation error)\n", texID);
    printf("Value at center:   %f\n", val_center);
    printf("Value at offset:   %f\n", val_offset);
    printf("Gradient check:    %f\n", (val_offset - val_center));
    printf("======================\n\n");
}


void savePPM_C(float* data, int width, int height, const char* filename) {
    FILE* fp = fopen(filename, "wb"); 
    if (!fp) {
        printf("Error opening file %s\n", filename);
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    unsigned char* pixelBuffer = (unsigned char*)malloc(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        float val = data[i];
        
        float norm = (val + 1.0f) * 0.5f; 

        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;

        unsigned char c = (unsigned char)(norm * 255.0f);
        
        pixelBuffer[i * 3 + 0] = c; // R
        pixelBuffer[i * 3 + 1] = c; // G
        pixelBuffer[i * 3 + 2] = c; // B
    }

    fwrite(pixelBuffer, sizeof(unsigned char), width * height * 3, fp);
    
    free(pixelBuffer);
    fclose(fp);
    printf("Saved %s\n", filename);
}



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


    float* d_noiseMap;
    size_t sizeBytes = NOISE_WIDTH * NOISE_HEIGHT * sizeof(float);
    cudaMalloc(&d_noiseMap, sizeBytes);

    dim3 noise_threads(16, 16);
    dim3 noise_blocks((NOISE_WIDTH + 15) / 16, (NOISE_HEIGHT + 15) / 16);

    int mySeed = (int)time(NULL);
    const unsigned long long seed = 1; //time(NULL);

    printf("Generating noise with Seed: %d\n", mySeed);

    generate_noise_kernel<<<noise_blocks, noise_threads>>>(d_noiseMap, NOISE_WIDTH, NOISE_HEIGHT, mySeed);
    cudaDeviceSynchronize();


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, NOISE_WIDTH, NOISE_HEIGHT);

    cudaMemcpy2DToArray(cuArray, 0, 0, d_noiseMap, 
                        NOISE_WIDTH * sizeof(float), NOISE_WIDTH * sizeof(float), 
                        NOISE_HEIGHT, cudaMemcpyDeviceToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t hostTexObj = 0;
    cudaCreateTextureObject(&hostTexObj, &resDesc, &texDesc, NULL);

    void* ptr_to_global_var;
    cudaGetSymbolAddress(&ptr_to_global_var, d_globalNoiseTex); 
    cudaMemcpy(ptr_to_global_var, &hostTexObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);


//    std::vector<float> h_noiseMap(WIDTH * HEIGHT);
//    cudaMemcpy(h_noiseMap.data(), d_noiseMap, sizeBytes, cudaMemcpyDeviceToHost);
//    savePPM(h_noiseMap.data(), WIDTH, HEIGHT, "noise_modern.ppm");
    debug_texture_check<<<1, 1>>>();
    cudaDeviceSynchronize(); 

    float* h_noiseMap = (float*)malloc(sizeBytes);
    
    if (h_noiseMap == NULL) {
        printf("Failed to allocate host memory!\n");
        return -1;
    }

    cudaMemcpy(h_noiseMap, d_noiseMap, sizeBytes, cudaMemcpyDeviceToHost);

    savePPM_C(h_noiseMap, NOISE_WIDTH, NOISE_HEIGHT, "noise_modern.ppm");

    free(h_noiseMap);



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
