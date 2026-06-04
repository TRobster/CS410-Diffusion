#ifndef KERNEL_CORRECTNESS_H
#define KERNEL_CORRECTNESS_H

#include <hip/hip_runtime.h>
#include "kernel_utils.h"
#include <utility>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

std::vector<float> generate_u(
    int                     rows,
    int                     cols,
    std::pair<float, float> range,
    float                   uniform_val = 0.0f
);

std::vector<float> heat_equation_neumann_cpu(
    const std::vector<float> u,
    std::vector<float>       u_new,
    int                      rows,
    int                      cols,
    float                    alpha
);

bool test_kernel_uniform_field(
    const std::vector<float> uniform_field,
    std::vector<float>               u_new,
    int                               rows,
    int                               cols,
    float                            alpha,
    std::string             kernel_version,
    int                             stride = 1,
    int                         block_size = 256,
    int                          grid_size = -1,
    access_distribution              reads = {}
);

bool test_kernel_cpu(
    const std::vector<float> input_field,
    std::vector<float>             d_u_new,
    std::vector<float>             h_u_new,
    int                               rows,
    int                               cols,
    float                            alpha,
    std::string             kernel_version,
    int                             stride = 1,
    int                         block_size = 256,
    int                          grid_size = -1,
    access_distribution              reads = {}
);

#endif
