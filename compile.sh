nvcc -O3 -rdc=true -dc main.cu -o main.o &
nvcc -O3 -rdc=true -dc kernels.cu -o kernels.o &
gcc  -O3 -c  helpers.c -o helpers.o &
wait
nvcc -O3 -rdc=true main.o kernels.o helpers.o -o simulation -lcudart
