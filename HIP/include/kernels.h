#ifndef KERNELS_H
#define KERNELS_H

#include <hip/hip_runtime.h>
#include <vector>

#define BLOCK_X 16
#define BLOCK_Y 16

// Single explicit-Euler step of the 2D heat equation u_t = kappa*(u_xx + u_yy).
// Neumann (zero-flux) boundary conditions: du/dn = 0 at all edges, implemented
// by clamping out-of-bounds indices to the nearest boundary cell.
// Shared memory tile of size (BLOCK_Y+2) x (BLOCK_X+2) is loaded per block.
//
// alpha = kappa * dt  (with dx = dy = 1 pixel spacing)
// Stability requires alpha <= 0.25.
void heat_equation_tunable_stride(
    const float* d_u,
    float*       d_u_new,
    int          rows,
    int          cols,
    float        alpha,
    int          stride,
    int          deviceId
);

//
void heat_equation_tunable_dimensions(
    const float* d_u,
    float*       d_u_new,
    int          rows,
    int          cols,
    float        alpha,
    int          block_size,
    int          grid_size,
    int          deviceId = -1
);

//
void heat_equation_adaptive(
    const float*         d_u,
    float*           d_u_new,
    int                 rows,
    int                 cols,
    float               alpha,
    access_distribution reads, 
    int         deviceId = -1
);

#endif