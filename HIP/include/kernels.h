#ifndef KERNELS_H
#define KERNELS_H

#include <hip/hip_runtime.h>
#include <vector>
#include "kernel_utils.h"

#define BLOCK_X 16
#define BLOCK_Y 16


__global__ void heat_equation_neumann_kernel(
    const float* __restrict__ u,
    float* __restrict__       u_new,
    int                       rows,
    int                       cols,
    float                     alpha,
    int                       stride
);


void kernel_wrapper_tunable_stride(
    const float* d_u,
    float*       d_u_new,
    int          rows,
    int          cols,
    float        alpha,
    int          stride,
    int          deviceId
);

//
void kernel_wrapper_tunable_dimensions(
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
void kernel_wrapper_adaptive(
    const float*         d_u,
    float*           d_u_new,
    int                 rows,
    int                 cols,
    float               alpha,
    access_distribution reads, 
    int         deviceId = -1
);

#endif