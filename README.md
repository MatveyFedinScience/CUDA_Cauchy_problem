# CUDA_Cauchy_problem
High efficient program for solving Cauchy problem on a circle

Currently, this program allows to track the evolution of a system of non-interacting particles launched at a some interval of speeds in different direction.

Finally, program generates image with (x,y) related to start angle and start relativ velocity angle. RGB pixel means (start speed, end speed, time in motion)

## USING

```bash
./compile.sh && ./simulation
```

V1.0 Changelog
1) Everything done, generation of arbitrary particles and simulation using the potential hardcoded in the potentials.cuh file.
