nvcc -O3 -dc main.cu -o main.o &
nvcc -O3 -dc kernels.cu -o kernels.o &
wait
nvcc -O3 main.o kernels.o -o simulation -lcudart
