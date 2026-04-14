# CUDA_Cauchy_problem

High efficient program for solving Cauchy problem on a circle

Currently, this program allows to track the evolution of a system of non-interacting particles launched at a some interval of speeds in different direction.

Finally, program generates image with (x,y) related to start angle and start relativ velocity angle. RGB pixel means (start speed, end speed, time in motion)

# USING

```bash
./compile.sh && ./simulation
```

## Command line options

- `--forces` : Use force field mode instead of potential. Generates independent Perlin noise for F_x and F_y components.
- `--file <filename>` : Load potential or force field from PPM image file instead of generating noise.

### Examples

```bash
# Default: generate Perlin noise potential, save as noise_modern.ppm
./simulation

# Use force field mode, generate F_x/F_y noise, save as forces_modern.ppm
./simulation --forces

# Load potential from existing PPM file
./simulation --file my_potential.ppm

# Load force field from PPM file (B=F_x, G=F_y)
./simulation --forces --file my_forces.ppm
```

## Image formats

- **Potential mode**: Grayscale PPM (R=G=B). Pixel value [0,255] maps to potential [-1,1].
- **Force field mode**: RGB PPM with B=F_x, G=F_y, R ignored. Each component maps [-1,1] → [0,255].

Generated images are saved as `noise_modern.ppm` (potential) or `forces_modern.ppm` (forces) when not using `--file`.

## V2.0 Changelog

*   Add potential from picture.

## V2.1 Changelog

*   Add `--forces` flag for force field simulation mode (2× faster than potential gradient)
*   Add `--file` flag to load potentials/forces from PPM image files
*   Generated images are not overwritten when loading from file

## V2.1 Changelog

*   Add `--forces` flag for force field simulation mode (2× faster than potential gradient)
*   Add `--file` flag to load potentials/forces from PPM image files
*   Generated images are not overwritten when loading from file

## Credits

This project uses the following open-source libraries:

*   **FastNoiseLiteCUDA** by [NeKon69](https://github.com/NeKon69/FastNoiseLiteCUDA) - CUDA port of FastNoiseLite.
*   **FastNoiseLite** by [Auburn](https://github.com/Auburn/FastNoiseLite) - The original noise generation library.
