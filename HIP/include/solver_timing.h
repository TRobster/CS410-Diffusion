#ifndef SOLVER_TIMING_H
#define SOLVER_TIMING_H

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

void clean_timing_data(
    std::vector<float> times,
    int                sd_tolerance
);

void time_solver(
    std::vector<float> u,
    float              alpha,
    int                rows,
    int                cols,
    int                iters,
    int                gridsize,
    int                blocksize,
    int                stride
);

#endif
