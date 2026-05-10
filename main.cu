#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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
__device__ cudaTextureObject_t d_FxTex;
__device__ cudaTextureObject_t d_FyTex;
__device__ bool g_use_forces = false; 



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

void savePPM_forces(float* Fx, float* Fy, int width, int height, const char* filename) {
    FILE* fp = fopen(filename, "wb"); 
    if (!fp) {
        printf("Error opening file %s\n", filename);
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    unsigned char* pixelBuffer = (unsigned char*)malloc(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        float fx = Fx[i];
        float fy = Fy[i];
        
        float norm_fx = (fx + 1.0f) * 0.5f; 
        float norm_fy = (fy + 1.0f) * 0.5f;

        if (norm_fx < 0.0f) norm_fx = 0.0f;
        if (norm_fx > 1.0f) norm_fx = 1.0f;
        if (norm_fy < 0.0f) norm_fy = 0.0f;
        if (norm_fy > 1.0f) norm_fy = 1.0f;

        unsigned char b = (unsigned char)(norm_fx * 255.0f);
        unsigned char g = (unsigned char)(norm_fy * 255.0f);
        
        pixelBuffer[i * 3 + 0] = 0;      // R
        pixelBuffer[i * 3 + 1] = g;      // G = F_y
        pixelBuffer[i * 3 + 2] = b;      // B = F_x
    }

    fwrite(pixelBuffer, sizeof(unsigned char), width * height * 3, fp);
    
    free(pixelBuffer);
    fclose(fp);
    printf("Saved %s (B=F_x, G=F_y)\n", filename);
}

__global__ void generate_force_kernel(float* output, int width, int height, int seed)
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

    float noiseValue = NOISE_SCALE * ( noise.GetNoise((float)x, (float)y) ) * IN_CIRCLE(X, Y);

    output[y * width + x] = noiseValue;
}

float* loadPPM(const char* filename, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        return NULL;
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        printf("Not a P6 PPM file: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d %d", width, height);
    int maxval;
    fscanf(fp, "%d", &maxval);
    fgetc(fp); // skip newline

    if (maxval != 255) {
        printf("Unsupported maxval %d (expected 255)\n", maxval);
        fclose(fp);
        return NULL;
    }

    size_t num_pixels = (*width) * (*height);
    unsigned char* rgb_data = (unsigned char*)malloc(num_pixels * 3);
    if (!rgb_data) {
        printf("Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    if (fread(rgb_data, 1, num_pixels * 3, fp) != num_pixels * 3) {
        printf("Failed to read pixel data\n");
        free(rgb_data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    // Convert RGB to grayscale float [-1,1]
    float* float_data = (float*)malloc(num_pixels * sizeof(float));
    if (!float_data) {
        printf("Memory allocation failed for float data\n");
        free(rgb_data);
        return NULL;
    }

    for (size_t i = 0; i < num_pixels; i++) {
        // Assuming grayscale PPM (R=G=B)
        unsigned char gray = rgb_data[i * 3];
        float_data[i] = (gray / 255.0f) * 2.0f - 1.0f; // [0,255] -> [-1,1]
    }

    free(rgb_data);
    printf("Loaded %s: %dx%d\n", filename, *width, *height);
    return float_data;
}

float* loadPPM_forces(const char* filename, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        return NULL;
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        printf("Not a P6 PPM file: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d %d", width, height);
    int maxval;
    fscanf(fp, "%d", &maxval);
    fgetc(fp); // skip newline

    if (maxval != 255) {
        printf("Unsupported maxval %d (expected 255)\n", maxval);
        fclose(fp);
        return NULL;
    }

    size_t num_pixels = (*width) * (*height);
    unsigned char* rgb_data = (unsigned char*)malloc(num_pixels * 3);
    if (!rgb_data) {
        printf("Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    if (fread(rgb_data, 1, num_pixels * 3, fp) != num_pixels * 3) {
        printf("Failed to read pixel data\n");
        free(rgb_data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    // Convert RGB to two float arrays Fx, Fy (interleaved in output)
    // Expecting B=F_x, G=F_y, R=ignored
    float* float_data = (float*)malloc(num_pixels * 2 * sizeof(float));
    if (!float_data) {
        printf("Memory allocation failed for float data\n");
        free(rgb_data);
        return NULL;
    }

    for (size_t i = 0; i < num_pixels; i++) {
        unsigned char b = rgb_data[i * 3 + 2]; // B = F_x
        unsigned char g = rgb_data[i * 3 + 1]; // G = F_y
        // R = rgb_data[i*3+0] ignored
        
        float_data[i * 2] = (b / 255.0f) * 2.0f - 1.0f; // F_x [-1,1]
        float_data[i * 2 + 1] = (g / 255.0f) * 2.0f - 1.0f; // F_y [-1,1]
    }

    free(rgb_data);
    printf("Loaded forces %s: %dx%d (B=F_x, G=F_y)\n", filename, *width, *height);
    return float_data;
}

int main(int argc, char** argv) {
    bool use_forces = false;
    const char* input_file = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--forces")) {
            use_forces = true;
        } else if (!strcmp(argv[i], "--file")) {
            if (i + 1 < argc) {
                input_file = argv[i + 1];
                i++; // skip filename
            } else {
                printf("Error: --file requires a filename argument\n");
                return 1;
            }
        }
    }
    printf("Forces mode: %s\n", use_forces ? "enabled" : "disabled");
    if (input_file) {
        printf("Input file: %s\n", input_file);
    } else {
        printf("Input: generated noise\n");
    }

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
    float* d_blurred;
    size_t sizeBytes = NOISE_WIDTH * NOISE_HEIGHT * sizeof(float);
    cudaMalloc(&d_noiseMap, sizeBytes);
    cudaMalloc(&d_blurred, sizeBytes);

    dim3 noise_threads(16, 16);
    dim3 noise_blocks((NOISE_WIDTH + 15) / 16, (NOISE_HEIGHT + 15) / 16);

    int mySeed = (int)time(NULL);
    const unsigned long long seed = 1; //time(NULL);

    printf("Generating noise with Seed: %d\n", mySeed);

    cudaTextureObject_t hostTexObj = 0;
    cudaArray* cuArray = NULL;
    cudaTextureObject_t hostFxObj = 0;
    cudaTextureObject_t hostFyObj = 0;
    cudaArray* cuArray_Fx = NULL;
    cudaArray* cuArray_Fy = NULL;
    float* d_Fx = NULL;
    float* d_Fy = NULL;
    float* h_noiseMap = NULL;

    if (!use_forces) {
        // Potential mode: load from file or generate
        if (input_file) {
            int loaded_width, loaded_height;
            float* h_loaded = loadPPM(input_file, &loaded_width, &loaded_height);
            if (!h_loaded) {
                printf("Failed to load potential from %s\n", input_file);
                return 1;
            }
            if (loaded_width != NOISE_WIDTH || loaded_height != NOISE_HEIGHT) {
                printf("Warning: loaded image size %dx%d doesn't match expected %dx%d\n",
                       loaded_width, loaded_height, (int)NOISE_WIDTH, (int)NOISE_HEIGHT);
                // We'll use it anyway, but texture coordinates assume NOISE_WIDTH/HEIGHT
            }
            // Copy loaded data to device
            CUDA_CHECK(cudaMemcpy(d_noiseMap, h_loaded, sizeBytes, cudaMemcpyHostToDevice));
            free(h_loaded);
            printf("Loaded potential from %s\n", input_file);
        } else {
            // Generate noise
            generate_noise_kernel<<<noise_blocks, noise_threads>>>(d_noiseMap, NOISE_WIDTH, NOISE_HEIGHT, mySeed);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

#if BLUR_ENABLED
        {
            dim3 blur_block(BLUR_BLOCK_DIM, BLUR_BLOCK_DIM);
            dim3 blur_grid((NOISE_WIDTH  + BLUR_BLOCK_DIM - 1) / BLUR_BLOCK_DIM,
                           (NOISE_HEIGHT + BLUR_BLOCK_DIM - 1) / BLUR_BLOCK_DIM);
            gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_noiseMap, d_blurred,
                                                             NOISE_WIDTH, NOISE_HEIGHT);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
#endif

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, NOISE_WIDTH, NOISE_HEIGHT));
        CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, d_blurred, 
                            NOISE_WIDTH * sizeof(float), NOISE_WIDTH * sizeof(float), 
                            NOISE_HEIGHT, cudaMemcpyDeviceToDevice));

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

        CUDA_CHECK(cudaCreateTextureObject(&hostTexObj, &resDesc, &texDesc, NULL));

        void* ptr_to_global_var;
        CUDA_CHECK(cudaGetSymbolAddress(&ptr_to_global_var, d_globalNoiseTex)); 
        CUDA_CHECK(cudaMemcpy(ptr_to_global_var, &hostTexObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

        debug_texture_check<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize()); 

        // Save noise image only if generated, not loaded
        if (!input_file) {
            h_noiseMap = (float*)malloc(sizeBytes);
            if (h_noiseMap == NULL) {
                printf("Failed to allocate host memory!\n");
                return -1;
            }
            CUDA_CHECK(cudaMemcpy(h_noiseMap, d_noiseMap, sizeBytes, cudaMemcpyDeviceToHost));
            savePPM_C(h_noiseMap, NOISE_WIDTH, NOISE_HEIGHT, "noise_modern.ppm");
            free(h_noiseMap);
        }
    } else {
        // Forces mode: load from file or generate
        CUDA_CHECK(cudaMalloc(&d_Fx, sizeBytes));
        CUDA_CHECK(cudaMalloc(&d_Fy, sizeBytes));

        if (input_file) {
            int loaded_width, loaded_height;
            float* h_loaded = loadPPM_forces(input_file, &loaded_width, &loaded_height);
            if (!h_loaded) {
                printf("Failed to load forces from %s\n", input_file);
                return 1;
            }
            if (loaded_width != NOISE_WIDTH || loaded_height != NOISE_HEIGHT) {
                printf("Warning: loaded image size %dx%d doesn't match expected %dx%d\n",
                       loaded_width, loaded_height, (int)NOISE_WIDTH, (int)NOISE_HEIGHT);
            }
            // h_loaded contains interleaved Fx,Fy,Fx,Fy,... 
            // Need to separate into d_Fx and d_Fy
            size_t num_pixels = loaded_width * loaded_height;
            float* h_Fx_buf = (float*)malloc(num_pixels * sizeof(float));
            float* h_Fy_buf = (float*)malloc(num_pixels * sizeof(float));
            if (!h_Fx_buf || !h_Fy_buf) {
                printf("Memory allocation failed\n");
                free(h_loaded);
                return 1;
            }
            for (size_t i = 0; i < num_pixels; i++) {
                h_Fx_buf[i] = h_loaded[i * 2];
                h_Fy_buf[i] = h_loaded[i * 2 + 1];
            }
            CUDA_CHECK(cudaMemcpy(d_Fx, h_Fx_buf, sizeBytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Fy, h_Fy_buf, sizeBytes, cudaMemcpyHostToDevice));
            free(h_Fx_buf);
            free(h_Fy_buf);
            free(h_loaded);
            printf("Loaded forces from %s\n", input_file);
        } else {
            // Generate noise forces
            generate_force_kernel<<<noise_blocks, noise_threads>>>(d_Fx, NOISE_WIDTH, NOISE_HEIGHT, mySeed);
            generate_force_kernel<<<noise_blocks, noise_threads>>>(d_Fy, NOISE_WIDTH, NOISE_HEIGHT, mySeed + 1);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Save generated forces image
            float* h_Fx = (float*)malloc(sizeBytes);
            float* h_Fy = (float*)malloc(sizeBytes);
            if (!h_Fx || !h_Fy) {
                printf("Failed to allocate host memory!\n");
                return -1;
            }
            CUDA_CHECK(cudaMemcpy(h_Fx, d_Fx, sizeBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_Fy, d_Fy, sizeBytes, cudaMemcpyDeviceToHost));
            savePPM_forces(h_Fx, h_Fy, NOISE_WIDTH, NOISE_HEIGHT, "forces_modern.ppm");
            free(h_Fx);
            free(h_Fy);
        }

#if BLUR_ENABLED
        {
            dim3 blur_block(BLUR_BLOCK_DIM, BLUR_BLOCK_DIM);
            dim3 blur_grid((NOISE_WIDTH  + BLUR_BLOCK_DIM - 1) / BLUR_BLOCK_DIM,
                           (NOISE_HEIGHT + BLUR_BLOCK_DIM - 1) / BLUR_BLOCK_DIM);

            gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_Fx, d_blurred,
                                                             NOISE_WIDTH, NOISE_HEIGHT);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            CUDA_CHECK(cudaMallocArray(&cuArray_Fx, &channelDesc, NOISE_WIDTH, NOISE_HEIGHT));
            CUDA_CHECK(cudaMallocArray(&cuArray_Fy, &channelDesc, NOISE_WIDTH, NOISE_HEIGHT));

            CUDA_CHECK(cudaMemcpy2DToArray(cuArray_Fx, 0, 0, d_blurred,
                                NOISE_WIDTH * sizeof(float), NOISE_WIDTH * sizeof(float),
                                NOISE_HEIGHT, cudaMemcpyDeviceToDevice));

            gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_Fy, d_blurred,
                                                             NOISE_WIDTH, NOISE_HEIGHT);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy2DToArray(cuArray_Fy, 0, 0, d_blurred,
                                NOISE_WIDTH * sizeof(float), NOISE_WIDTH * sizeof(float),
                                NOISE_HEIGHT, cudaMemcpyDeviceToDevice));
        }
#else
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CUDA_CHECK(cudaMallocArray(&cuArray_Fx, &channelDesc, NOISE_WIDTH, NOISE_HEIGHT));
        CUDA_CHECK(cudaMallocArray(&cuArray_Fy, &channelDesc, NOISE_WIDTH, NOISE_HEIGHT));

        CUDA_CHECK(cudaMemcpy2DToArray(cuArray_Fx, 0, 0, d_Fx,
                            NOISE_WIDTH * sizeof(float), NOISE_WIDTH * sizeof(float),
                            NOISE_HEIGHT, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2DToArray(cuArray_Fy, 0, 0, d_Fy,
                            NOISE_WIDTH * sizeof(float), NOISE_WIDTH * sizeof(float),
                            NOISE_HEIGHT, cudaMemcpyDeviceToDevice));
#endif

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        resDesc.res.array.array = cuArray_Fx;
        CUDA_CHECK(cudaCreateTextureObject(&hostFxObj, &resDesc, &texDesc, NULL));
        resDesc.res.array.array = cuArray_Fy;
        CUDA_CHECK(cudaCreateTextureObject(&hostFyObj, &resDesc, &texDesc, NULL));

        // Bind textures to device symbols
        void* ptr;
        CUDA_CHECK(cudaGetSymbolAddress(&ptr, d_FxTex));
        CUDA_CHECK(cudaMemcpy(ptr, &hostFxObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGetSymbolAddress(&ptr, d_FyTex));
        CUDA_CHECK(cudaMemcpy(ptr, &hostFyObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

        // Enable forces mode
        bool forces_flag = true;
        CUDA_CHECK(cudaGetSymbolAddress(&ptr, g_use_forces));
        CUDA_CHECK(cudaMemcpy(ptr, &forces_flag, sizeof(bool), cudaMemcpyHostToDevice));
    }



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

    // Cleanup texture resources
    if (!use_forces) {
        if (hostTexObj != 0) CUDA_CHECK(cudaDestroyTextureObject(hostTexObj));
        if (cuArray != NULL) CUDA_CHECK(cudaFreeArray(cuArray));
    } else {
        if (hostFxObj != 0) CUDA_CHECK(cudaDestroyTextureObject(hostFxObj));
        if (hostFyObj != 0) CUDA_CHECK(cudaDestroyTextureObject(hostFyObj));
        if (cuArray_Fx != NULL) CUDA_CHECK(cudaFreeArray(cuArray_Fx));
        if (cuArray_Fy != NULL) CUDA_CHECK(cudaFreeArray(cuArray_Fy));
        if (d_Fx != NULL) CUDA_CHECK(cudaFree(d_Fx));
        if (d_Fy != NULL) CUDA_CHECK(cudaFree(d_Fy));
    }
    // d_noiseMap is always allocated at the beginning
    CUDA_CHECK(cudaFree(d_noiseMap));
    CUDA_CHECK(cudaFree(d_blurred));

    return 0;
}
