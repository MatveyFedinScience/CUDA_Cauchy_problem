#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"


#ifdef __cplusplus
extern "C" {
#endif


void init_particles_to_image(
    Particle* particles,
    int n,
    char** out_image
);

void final_particles_to_image(
    Particle* particles,
    int n,
    char** out_image
);

void save_single_image(
    const char* filename,
    char* image,
    int width,
    int height
);


#ifdef __cplusplus
}
#endif
