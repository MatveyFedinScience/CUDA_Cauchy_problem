#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "FastNoiseLiteCUDA.h" // Ваше правильное название

// ==========================================
// 1. СОВРЕМЕННЫЙ СПОСОБ: Глобальная переменная для объекта
// ==========================================
// Вместо texture<...> объявляем глобальную переменную на устройстве,
// которая будет хранить "handle" (дескриптор) нашей текстуры.
__device__ cudaTextureObject_t d_globalNoiseTex;

const int WIDTH = 1024;
const int HEIGHT = 1024;

// ==========================================
// 2. ЯДРО ГЕНЕРАЦИИ (Без изменений)
// ==========================================
__global__ void generate_noise_kernel(float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    FastNoiseLite noise(1337); 
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetFrequency(0.0025f);

    float noiseValue = noise.GetNoise((float)x, (float)y);
    output[y * width + x] = noiseValue;
}

// ==========================================
// 3. ФУНКЦИЯ POTENTIAL (Обновленная)
// ==========================================
__device__ __forceinline__ float potential(float x, float y) {
    // Мы обращаемся к глобальной переменной d_globalNoiseTex.
    // tex2D<float> - это перегрузка для Texture Objects.
    // +0.5f - сдвиг к центру пикселя для точности
    return tex2D<float>(d_globalNoiseTex, x + 0.5f, y + 0.5f);
}

// ==========================================
// 4. ТЕСТОВЫЙ КЕРНЕЛ
// ==========================================
__global__ void simulation_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 0) return; 

    float val1 = potential(10.0f, 10.0f);
    float val2 = potential(10.5f, 10.0f);

    printf("GPU Test (Texture Object):\n");
    printf("Pos (10.0, 10.0) = %f\n", val1);
    printf("Pos (10.5, 10.0) = %f (Interpolated)\n", val2);
}

// ==========================================
// 5. СОХРАНЕНИЕ PPM (Без изменений)
// ==========================================
void savePPM(const float* h_data, int width, int height, const char* filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return;
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        float val = h_data[i];
        float norm = (val + 1.0f) * 0.5f;
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;
        unsigned char pixel = static_cast<unsigned char>(norm * 255.0f);
        ofs << pixel << pixel << pixel;
    }
    ofs.close();
}

// ==========================================
// MAIN
// ==========================================
int main() {
    float* d_noiseMap;
    size_t sizeBytes = WIDTH * HEIGHT * sizeof(float);
    cudaMalloc(&d_noiseMap, sizeBytes);

    // 1. Генерируем шум
    dim3 threads(16, 16);
    dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    generate_noise_kernel<<<blocks, threads>>>(d_noiseMap, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Сохраняем для проверки
    std::vector<float> h_noiseMap(WIDTH * HEIGHT);
    cudaMemcpy(h_noiseMap.data(), d_noiseMap, sizeBytes, cudaMemcpyDeviceToHost);
    savePPM(h_noiseMap.data(), WIDTH, HEIGHT, "noise_modern.ppm");

    // ==========================================
    // 2. СОЗДАНИЕ TEXTURE OBJECT (Современный API)
    // ==========================================
    
    // А) Описываем ресурс (где лежат данные)
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_noiseMap;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32 бита на float
    resDesc.res.linear.sizeInBytes = sizeBytes;

    // Б) Описываем параметры текстуры (фильтрация, границы)
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType; // Читать как float
    
    // Включаем линейную интерполяцию (сглаживание)
    texDesc.filterMode = cudaFilterModeLinear; 
    
    // Обработка краев (Clamp - растягивать последний пиксель)
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    
    // Используем обычные координаты (0..WIDTH), а не (0..1)
    texDesc.normalizedCoords = 0;

    // В) Создаем объект на хосте
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Г) КОПИРУЕМ ОБЪЕКТ В ГЛОБАЛЬНУЮ ПЕРЕМЕННУЮ
    // Это ключевой момент. Мы передаем handle текстуры в __device__ переменную.
    cudaMemcpyToSymbol(d_globalNoiseTex, &texObj, sizeof(cudaTextureObject_t));

    // ==========================================
    
    // 3. Запускаем симуляцию
    // Аргументы передавать не нужно, ядро возьмет d_globalNoiseTex само
    simulation_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // 4. Уборка
    cudaDestroyTextureObject(texObj);
    cudaFree(d_noiseMap);

    return 0;
}
