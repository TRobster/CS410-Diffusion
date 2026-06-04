#include "heat_equation.h"
#include "kernels.h"
#include "kernel_utils.h"

using namespace std;

vector<float> heat_equation_2d(
    vector<float> d_u,
    int           rows,
    int           cols,
    float         alpha,
    int           iter,
    int           gridsize,
    int           blocksize,
    int           stride
) {
    int deviceId = select_device();
    HIP_CHECK(hipSetDevice(deviceId));
    size_t bytes = rows * cols * sizeof(float);

    float* dev_u     = nullptr;
    float* dev_u_new = nullptr;
    HIP_CHECK(hipMalloc(&dev_u,     bytes));
    HIP_CHECK(hipMalloc(&dev_u_new, bytes));
    HIP_CHECK(hipMemcpy(dev_u,     d_u.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_u_new, d_u.data(), bytes, hipMemcpyHostToDevice));

    dim3 block;
    dim3 grid;
    int kernel_stride;

    if (gridsize != -1) {
        auto [bx, by] = determine_block_dimensions(blocksize);
        block         = dim3(bx, by);
        grid          = dim3(gridsize, 1);
        kernel_stride = (rows * cols) / (blocksize * gridsize);
    } else {
        const int num_tiles_x = (cols + BLOCK_X - 1) / BLOCK_X;
        const int num_tiles_y = (rows + BLOCK_Y - 1) / BLOCK_Y;
        const int total_tiles = num_tiles_x * num_tiles_y;
        block         = dim3(BLOCK_X, BLOCK_Y);
        grid          = dim3((total_tiles + stride - 1) / stride, 1);
        kernel_stride = stride;
    }

    for (int i = 0; i < iter; i++) {
        hipLaunchKernelGGL(heat_equation_neumann_kernel, grid, block, 0, 0,
                           dev_u, dev_u_new, rows, cols, alpha, kernel_stride);
        HIP_CHECK(hipGetLastError());
        swap(dev_u, dev_u_new);
    }

    HIP_CHECK(hipDeviceSynchronize());

    vector<float> result(rows * cols);
    HIP_CHECK(hipMemcpy(result.data(), dev_u, bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dev_u));
    HIP_CHECK(hipFree(dev_u_new));

    return result;
}
