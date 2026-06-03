#include "correctness.h"
#include "kernels.h"
#include "kernel_utils.h"
#include <random>
#include <cmath>

std::vector<float> generate_u(
    int                      rows,
    int                      cols,
    std::pair<float, float>  range,
    float                    uniform_val
) 
{
    std::vector<float> u(rows * cols);
    if (uniform_val != 0.0f) {
        std::fill(u.begin(), u.end(), uniform_val);
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(range.first, range.second);
        for (auto& v : u) v = dist(rng);
    }
    return u;
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

bool test_kernel_uniform_field(
    const std::vector<float> uniform_field,
    std::vector<float>       u_new,
    int                      rows,
    int                      cols,
    float                    alpha,
    std::string              kernel_version
) 
{
    // A uniform field is a fixed point of the heat equation: one GPU step must
    // leave every cell at the original constant value.
    int deviceId = select_device();
    size_t bytes = rows * cols * sizeof(float);

    float* d_u     = nullptr;
    float* d_u_new = nullptr;
    HIP_CHECK(hipMalloc(&d_u,     bytes));
    HIP_CHECK(hipMalloc(&d_u_new, bytes));
    HIP_CHECK(hipMemcpy(d_u,     uniform_field.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u_new, u_new.data(),         bytes, hipMemcpyHostToDevice));

    // "default" is the only kernel version currently
    heat_equation_neumann_step(d_u, d_u_new, rows, cols, alpha, 1, deviceId);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(u_new.data(), d_u_new, bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_u_new));

    const float ref_val = uniform_field[0];
    const float tol     = 1e-5f;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(u_new[i] - ref_val) > tol) {
            std::cout << "FAILED uniform-field test at index " << i
                      << ": expected " << ref_val << ", got " << u_new[i] << "\n";
            return false;
        }
    }
    std::cout << "PASSED uniform-field test\n";
    return true;
}

bool test_kernel_cpu(
    const std::vector<float> input_field,
    std::vector<float>       d_u_new,
    std::vector<float>       h_u_new,
    int                      rows,
    int                      cols,
    float                    alpha,
    std::string              kernel_version
) 
{
    int deviceId = select_device();
    size_t bytes = rows * cols * sizeof(float);

    // Run GPU kernel and copy result back into d_u_new
    float* d_u      = nullptr;
    float* d_u_new_dev = nullptr;
    HIP_CHECK(hipMalloc(&d_u,        bytes));
    HIP_CHECK(hipMalloc(&d_u_new_dev, bytes));
    HIP_CHECK(hipMemcpy(d_u,         input_field.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u_new_dev, d_u_new.data(),     bytes, hipMemcpyHostToDevice));

    // "default" is the only kernel version currently
    heat_equation_neumann_step(d_u, d_u_new_dev, rows, cols, alpha, 1, deviceId);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(d_u_new.data(), d_u_new_dev, bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_u_new_dev));

    // Run CPU reference
    h_u_new = heat_equation_neumann_cpu(input_field, h_u_new, rows, cols, alpha);

    const float tol = 1e-4f;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(d_u_new[i] - h_u_new[i]) > tol) {
            std::cout << "FAILED CPU-vs-GPU test at index " << i
                      << ": CPU=" << h_u_new[i] << ", GPU=" << d_u_new[i] << "\n";
            return false;
        }
    }
    std::cout << "PASSED CPU-vs-GPU test\n";
    return true;
}
