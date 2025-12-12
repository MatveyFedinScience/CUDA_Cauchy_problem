#ifndef CONFIG_H
#define CONFIG_H

#define N_PARTICLES 10240
#define DT          0.001f
#define N_STEPS     10000
#define R_CIRCLE    1.0f
#define BLOCK_SIZE  256
#define IN_CIRCLE(x, y) ((int) ((x) * (x) + (y) * (y) < R_CIRCLE * R_CIRCLE))
#define CALC_METHOD gauss_legendre_step //available list: rk4_step, gauss_legendre_step

#endif
