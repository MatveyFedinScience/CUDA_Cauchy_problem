nvcc -O3 -dc main.cu -o main.o &
nvcc -O3 -dc kernels.cu -o kernels.o &
gcc  -O3 -c  helpers.c -o helpers.o &
wait
nvcc -O3 main.o kernels.o helpers.o -o simulation -lcudart
