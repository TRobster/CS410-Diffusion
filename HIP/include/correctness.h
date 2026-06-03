#ifndef CORRECTNESS_H
#define CORRECTNESS_H

#include <hip/hip_runtime.h>
#include <utility>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <vector>
#include <string>

std::vector<float> generate_u(
    int                      rows,
    int                      cols,
    std::pair<float, float> range,
    float uniform_val =       0.0
);

std::vector<float> heat_equation_neumann_cpu(
    const std::vector<float> u,
    std::vector<float>   u_new,
    int                   rows,
    int                   cols,
    float                alpha
);

bool test_kernel_uniform_field(
    const std::vector<float> uniform_field,
    std::vector<float>               u_new,
    int                               rows,
    int                               cols,
    float                            alpha,
    std::string             kernel_version
);

bool test_kernel_cpu(
    const std::vector<float> input_field,
    std::vector<float>             d_u_new,
    std::vector<float>             h_u_new,
    int                               rows,
    int                               cols,
    float                            alpha,
    std::string             kernel_version
);

#endif