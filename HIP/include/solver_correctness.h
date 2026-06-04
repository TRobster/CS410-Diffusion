#ifndef SOLVER_CORRECTNESS_H
#define SOLVER_CORRECTNESS_H

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

bool test_solver_uniform_field(
    const std::vector<float> uniform_field,
    int                      rows,
    int                      cols,
    float                    alpha,
    int                      iters,
    int                      gridsize,
    int                      blocksize,
    int                      stride
);

bool test_solver_cpu(
    const std::vector<float> input_field,
    int                      rows,
    int                      cols,
    float                    alpha,
    int                      iters,
    int                      gridsize,
    int                      blocksize,
    int                      stride
);

#endif
