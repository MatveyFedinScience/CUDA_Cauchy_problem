# CUDA_Cauchy_problem

High efficient program for solving Cauchy problem on a circle

Currently, this program allows to track the evolution of a system of non-interacting particles launched at a some interval of speeds in different direction.

Finally, program generates image with (x,y) related to start angle and start relativ velocity angle. RGB pixel means (start speed, end speed, time in motion)

<<<<<<< HEAD
=======

>>>>>>> 30010a7 (version 2.0)
## USING

```bash
./compile.sh && ./simulation
```

<<<<<<< HEAD
V1.0 Changelog
1) Everything done, generation of arbitrary particles and simulation using the potential hardcoded in the potentials.cuh file.
=======
## V2.0 Changelog

*   Add potential from picture.

## Credits

This project uses the following open-source libraries:

*   **FastNoiseLiteCUDA** by [NeKon69](https://github.com/NeKon69/FastNoiseLiteCUDA) - CUDA port of FastNoiseLite.
*   **FastNoiseLite** by [Auburn](https://github.com/Auburn/FastNoiseLite) - The original noise generation library.
>>>>>>> 30010a7 (version 2.0)
