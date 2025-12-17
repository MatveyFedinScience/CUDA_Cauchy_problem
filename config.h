#ifndef CONFIG_H
#define CONFIG_H

/*
it's configured for
Device: NVIDIA GeForce RTX 4060 Ti 16Gb
SM count: 34
Max threads per SM: 1536
maxThreadsPerBlock = 1024
maxThreadsDim      = 1024 1024 64
maxGridSize        = 2147483647 65535 65535
*/
#define N_PARTICLES (1024*16*4) //1024*8 per one GPU cycle
#define DT          0.0001f
#define N_STEPS     100000
#define R_CIRCLE    1.0f
#define BLOCK_SIZE  256
#define IN_CIRCLE(x, y)         ((int) ((x) * (x) + (y) * (y) < R_CIRCLE * R_CIRCLE))
#define SMOOTH_IN_CIRCLE(x, y) \
            ((float) (1.0f / (1.0f + expf(((x) * (x) + (y) * (y) - R_CIRCLE * R_CIRCLE))*10.0f)))
#define CALC_METHOD rk4_step //available list: rk4_step, gauss_legendre_step
#define MAX_INIT_VEL .5f
#define MIN_INIT_VEL .5f

#endif
