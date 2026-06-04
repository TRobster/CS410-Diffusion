#include "kernel_correctness.h"
#include "kernels.h"
#include "kernel_utils.h"
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

static void dispatch_kernel(
    const float*        d_u,
    float*              d_u_new,
    int                 rows,
    int                 cols,
    float               alpha,
    const std::string&  kernel_version,
    int                 stride,
    int                 block_size,
    int                 grid_size,
    access_distribution reads,
    int                 deviceId
) {
    if (kernel_version == "stride") {
        kernel_wrapper_tunable_stride(d_u, d_u_new, rows, cols, alpha, stride, deviceId);
    } else if (kernel_version == "dimension") {
        kernel_wrapper_tunable_dimensions(d_u, d_u_new, rows, cols, alpha, block_size, grid_size, deviceId);
    } else if (kernel_version == "adaptive") {
        kernel_wrapper_adaptive(d_u, d_u_new, rows, cols, alpha, reads, deviceId);
    } else {
        kernel_wrapper_tunable_stride(d_u, d_u_new, rows, cols, alpha, 1, deviceId);
    }
}

bool test_kernel_uniform_field(
    const std::vector<float> uniform_field,
    std::vector<float>       u_new,
    int                      rows,
    int                      cols,
    float                    alpha,
    std::string              kernel_version,
    int                      stride,
    int                      block_size,
    int                      grid_size,
    access_distribution      reads
) {
    int deviceId = select_device();
    size_t bytes = rows * cols * sizeof(float);

    float* d_u     = nullptr;
    float* d_u_new = nullptr;
    HIP_CHECK(hipMalloc(&d_u,     bytes));
    HIP_CHECK(hipMalloc(&d_u_new, bytes));
    HIP_CHECK(hipMemcpy(d_u,     uniform_field.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u_new, u_new.data(),         bytes, hipMemcpyHostToDevice));

    dispatch_kernel(d_u, d_u_new, rows, cols, alpha,
                    kernel_version, stride, block_size, grid_size, reads, deviceId);
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
    std::string              kernel_version,
    int                      stride,
    int                      block_size,
    int                      grid_size,
    access_distribution      reads
) {
    int deviceId = select_device();
    size_t bytes = rows * cols * sizeof(float);

    float* d_u         = nullptr;
    float* d_u_new_dev = nullptr;
    HIP_CHECK(hipMalloc(&d_u,         bytes));
    HIP_CHECK(hipMalloc(&d_u_new_dev, bytes));
    HIP_CHECK(hipMemcpy(d_u,         input_field.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u_new_dev, d_u_new.data(),     bytes, hipMemcpyHostToDevice));

    dispatch_kernel(d_u, d_u_new_dev, rows, cols, alpha,
                    kernel_version, stride, block_size, grid_size, reads, deviceId);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(d_u_new.data(), d_u_new_dev, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_u_new_dev));

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
