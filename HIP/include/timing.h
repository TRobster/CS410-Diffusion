#ifndef TIMING_H
#define TIMING_H

#include <hip/hip_runtime.h>
#include <utility>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <vector>
#include <string>


void clean_timing_data(
    std::vector<float> times,
    int         sd_tolerance
);

void time_kernel(
    std::vector<float>       u,
    float                alpha,
    int                   rows,
    int                   cols,
    int          kernel_stride,
    int             iterations,
    std::string kernel_version
);

#endif