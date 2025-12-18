#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "config.h"

void init_particles_to_image(
    Particle* particles,
    int n,
    char** out_image
)
{
    int S = (int)ceilf(sqrtf((float)n));

    unsigned char* img =
        (unsigned char*)malloc(S * S * 3);

    memset(img, 0, S * S * 3);

    for (int i = 0; i < n; ++i) {
        int y = i / S;
        int x = i % S;

        float vx = particles[i].vx;
        float vy = particles[i].vy;

        float speed = sqrtf(vx * vx + vy * vy)/MAX_INIT_VEL*255;

        if (speed > 255.0f) speed = 255.0f;
        if (speed < 0.0f)   speed = 0.0f;

        unsigned char c = (unsigned char)speed;

        int idx = (y * S + x) * 3;
        img[idx + 0] = c;   /* R */
        img[idx + 1] = 0;   /* G */
        img[idx + 2] = 0;   /* B */
    }

    *out_image = (char*)img;
}


void final_particles_to_image(
    Particle* particles,
    int n,
    char** out_image
)
{
    int S = (int)ceilf(sqrtf((float)n));

    for (int i = 0; i < n; ++i) {
        int y = i / S;
        int x = i % S;

        float vx = particles[i].vx;
        float vy = particles[i].vy;
        int steps = 255 * particles[i].steps / N_STEPS;
        float speed = sqrtf(vx * vx + vy * vy) / MAX_INIT_VEL * 255;

        if (speed > 255.0f) speed = 255.0f;
        if (speed < 0.0f)   speed = 0.0f;

        unsigned char c = (unsigned char)speed;

        int idx = (y * S + x) * 3;
        (*out_image)[idx + 1] = c;       /* G */
        (*out_image)[idx + 2] = steps;   /* B */
    }

}


void save_single_image(
    const char* filename,
    char* image,
    int width,
    int height
)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    fwrite(image, 1, width * height * 3, fp);

    fclose(fp);
}

