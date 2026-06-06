#include "solver_correctness.h"
#include "heat_equation.h"
#include <cstdio>
#include <cstdlib>
#include <utility>   
#include <random>
#include <cmath>

std::vector<float> generate_u(
    int                     rows,
    int                     cols,
    std::pair<float, float> range,
    float                   uniform_val
) {
    std::vector<float> img(rows * cols);
    
    for (int i = 0; i < rows * cols; ++i) img[i] = 0.0f;   // black

    // Centered bright square.
    for (int row = cols / 4; row < 3 * cols / 4; ++row)
        for (int col = rows / 4; col < 3 * rows / 4; ++col)
            img[row * rows + col] = 255.0f;

    // A few single-pixel bright points: {row, col}.
    const int pts[][2] = { {64, 64}, {64, 448}, {448, 64}, {448, 448} };
    for (const auto& p : pts)
        img[p[0] * rows + p[1]] = 255.0f;

    return img;
}

std::vector<float> heat_equation_neumann_cpu(
    const std::vector<float> u,
    std::vector<float>       u_new,
    int                      rows,
    int                      cols,
    float                    alpha
) {
    auto clamp_row = [&](int r) { return r < 0 ? 0 : (r >= rows ? rows - 1 : r); };
    auto clamp_col = [&](int c) { return c < 0 ? 0 : (c >= cols ? cols - 1 : c); };

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float center = u[r * cols + c];
            float left   = u[r * cols + clamp_col(c - 1)];
            float right  = u[r * cols + clamp_col(c + 1)];
            float above  = u[clamp_row(r - 1) * cols + c];
            float below  = u[clamp_row(r + 1) * cols + c];
            u_new[r * cols + c] = center + alpha * (left + right + above + below - 4.0f * center);
        }
    }
    return u_new;
}

bool test_solver_uniform_field(
    const std::vector<float> uniform_field,
    int                      rows,
    int                      cols,
    float                    alpha,
    int                      iters,
    int                      gridsize,
    int                      blocksize,
    int                      stride
) {
    std::vector<float> result = heat_equation_2d(
        uniform_field, rows, cols, alpha, iters, gridsize, blocksize, stride);

    const float ref_val = uniform_field[0];
    const float tol     = 1e-5f;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(result[i] - ref_val) > tol) {
            std::cout << "FAILED uniform-field test at index " << i
                      << ": expected " << ref_val << ", got " << result[i] << "\n";
            return false;
        }
    }
    std::cout << "PASSED uniform-field test\n";
    return true;
}

bool test_solver_cpu(
    const std::vector<float> input_field,
    int                      rows,
    int                      cols,
    float                    alpha,
    int                      iters,
    int                      gridsize,
    int                      blocksize,
    int                      stride
) {
    std::vector<float> gpu_result = heat_equation_2d(
        input_field, rows, cols, alpha, iters, gridsize, blocksize, stride);

    std::vector<float> cpu_u     = input_field;
    std::vector<float> cpu_u_new(rows * cols, 0.0f);
    for (int i = 0; i < iters; i++) {
        cpu_u_new = heat_equation_neumann_cpu(cpu_u, cpu_u_new, rows, cols, alpha);
        std::swap(cpu_u, cpu_u_new);
    }

    const float tol = 1e-3f;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(gpu_result[i] - cpu_u[i]) > tol) {
            std::cout << "FAILED CPU-vs-solver test at index " << i
                      << ": CPU=" << cpu_u[i] << ", GPU=" << gpu_result[i] << "\n";
            return false;
        }
    }

    return true;
}
