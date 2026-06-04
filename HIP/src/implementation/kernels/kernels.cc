#include <cmath>
#include "kernels.h"
#include "kernel_utils.h"

// 5-point finite-difference stencil for the 2D heat equation.
// Each block loads a (BLOCK_Y+2) x (BLOCK_X+2) shared-memory tile that
// includes a 1-cell halo on every side.  Neumann BCs are enforced by
// clamping the global (row, col) index before the global-memory read, so
// boundary cells see their own value as the out-of-bounds neighbor (zero
// normal derivative).
__global__ void heat_equation_neumann_kernel(
    const float* __restrict__ u,
    float* __restrict__       u_new,
    int                       rows,
    int                       cols,
    float                     alpha,
    int                       stride
)
{
    // Shared tile — declared once and reused across every tile this block processes.
    __shared__ float tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx         = threadIdx.x;
    const int ty         = threadIdx.y;
    const int linear_tid = ty * blockDim.x + tx;

    // Clamp indices for Neumann BC (zero-flux mirror).
    auto clamp_col = [&](int c) { return c < 0 ? 0 : (c >= cols ? cols - 1 : c); };
    auto clamp_row = [&](int r) { return r < 0 ? 0 : (r >= rows ? rows - 1 : r); };

    constexpr int TILE_SIZE   = (BLOCK_X + 2) * (BLOCK_Y + 2);
    const int     num_tiles_x = (cols + BLOCK_X - 1) / BLOCK_X;
    const int     total_tiles = num_tiles_x * ((rows + BLOCK_Y - 1) / BLOCK_Y);

    // Each block owns a contiguous chunk of stride tiles, so consecutive
    // iterations share halo data already warm in L2.
    //   block 0 → tiles [0,          stride)
    //   block 1 → tiles [stride,     2*stride)
    //   ...
    const int t_start = blockIdx.x * stride;
    const int t_end   = min(t_start + stride, total_tiles);

    for (int t = t_start; t < t_end; t++) {
        const int by       = t / num_tiles_x;
        const int bx       = t % num_tiles_x;
        const int base_row = by * BLOCK_Y;
        const int base_col = bx * BLOCK_X;
        const int gx       = base_col + tx;
        const int gy       = base_row + ty;

        // Load the (BLOCK_Y+2)×(BLOCK_X+2) halo tile from global memory.
        // All threads stay active; each handles at most 2 tile elements.
        for (int flat = linear_tid; flat < TILE_SIZE; flat += blockDim.x * blockDim.y) {
            const int tr = flat / (BLOCK_X + 2);
            const int tc = flat % (BLOCK_X + 2);
            tile[tr][tc] = u[clamp_row(base_row - 1 + tr) * cols
                             + clamp_col(base_col - 1 + tc)];
        }

        // Tile must be fully loaded before any thread reads it.
        __syncthreads();

        // Apply the 5-point stencil for valid grid cells.
        if (gx < cols && gy < rows) {
            const float center = tile[ty + 1][tx + 1];
            const float left   = tile[ty + 1][tx    ];
            const float right  = tile[ty + 1][tx + 2];
            const float above  = tile[ty    ][tx + 1];
            const float below  = tile[ty + 2][tx + 1];
            u_new[gy * cols + gx] = center + alpha * (left + right + above + below - 4.0f * center);
        }

        // All threads must finish reading tile before the next iteration
        // overwrites it with a new tile's data.
        __syncthreads();
    }
}

//
void kernel_wrapper_tunable_stride(
    const float* d_u,
    float*       d_u_new,
    int          rows,
    int          cols,
    float        alpha,
    int          stride,
    int          deviceId
)
{
    if (deviceId < 0) deviceId = select_device();
    HIP_CHECK(hipSetDevice(deviceId));

    dim3 block(BLOCK_X, BLOCK_Y);

    // stride = target number of tiles per block.
    // We therefore need ceil(total_tiles / stride) blocks to cover everything.
    const int num_tiles_x = (cols + BLOCK_X - 1) / BLOCK_X;
    const int num_tiles_y = (rows + BLOCK_Y - 1) / BLOCK_Y;
    const int total_tiles = num_tiles_x * num_tiles_y;
    const int num_blocks  = (total_tiles + stride - 1) / stride;
    dim3 grid(num_blocks, 1);

    hipLaunchKernelGGL(heat_equation_neumann_kernel, grid, block, 0, 0,
                       d_u, d_u_new, rows, cols, alpha, stride);
    HIP_CHECK(hipGetLastError());
}

//
void kernel_wrapper_tunable_dimensions(
    const float* d_u,
    float*       d_u_new,
    int          rows,
    int          cols,
    float        alpha,
    int          block_size,
    int          grid_size,
    int           deviceId
)
{
    if (deviceId < 0) deviceId = select_device();
    HIP_CHECK(hipSetDevice(deviceId));

    std::pair<int, int> dimensions = determine_block_dimensions(block_size);
    const int block_x = dimensions.first, block_y = dimensions.second;

    dim3 block(block_x, block_y);
    dim3 grid(grid_size, 1);

    // stride = target number of tiles per block.
    // We therefore need ceil(total_tiles / stride) blocks to cover everything.
    const int num_tiles_x = (cols + block_x - 1) / block_x;
    const int num_tiles_y = (rows + block_y - 1) / block_y;
    const int total_tiles = num_tiles_x * num_tiles_y;
    const int stride = (rows * cols) / (block_size * grid_size);
    const int num_blocks  = (total_tiles + stride - 1) / stride;

    hipLaunchKernelGGL(heat_equation_neumann_kernel, grid, block, 0, 0,
                       d_u, d_u_new, rows, cols, alpha, stride);
    HIP_CHECK(hipGetLastError());
}

//
void kernel_wrapper_adaptive(
    const float*         d_u,
    float*           d_u_new,
    int                 rows,
    int                 cols,
    float               alpha,
    access_distribution reads, 
    int              deviceId
)
{
    if (deviceId < 0) deviceId = select_device();
    HIP_CHECK(hipSetDevice(deviceId));

    int ideal_occupancy = determine_ideal_cu_occupancy(reads);
    int du_size = rows * cols;
    std::pair<int, int> dims = determine_dimensions_occupancy(du_size, BLOCK_X * BLOCK_Y, ideal_occupancy);
    int grid_size = dims.first, block_size = dims.second;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(grid_size, 1);

    // stride = target number of tiles per block.
    // We therefore need ceil(total_tiles / stride) blocks to cover everything.
    const int num_tiles_x = (cols + BLOCK_X - 1) / BLOCK_X;
    const int num_tiles_y = (rows + BLOCK_Y - 1) / BLOCK_Y;
    const int total_tiles = num_tiles_x * num_tiles_y;
    const int stride = du_size / (block_size * grid_size);
    const int num_blocks  = (total_tiles + stride - 1) / stride;

    hipLaunchKernelGGL(heat_equation_neumann_kernel, grid, block, 0, 0,
                       d_u, d_u_new, rows, cols, alpha, stride);
    HIP_CHECK(hipGetLastError());
}