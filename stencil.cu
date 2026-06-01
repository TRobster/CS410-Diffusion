#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define K 32
#define TPB 512
// RAD = ceil(N / 2) where N is the total order N'th derivative. 
#define RAD 1 // <-- number of halo cells needed for the boundary conditions of a given stencil. For central difference 1D, simply 1 neighbor each side. 

static int ts = TPB * K; // << -- STATIC Int is only accessible on CPU! Not GPU. #FIX 2
__global__ void memTest(int *x, int size) // Pass CPU integer as size. #FIX 2
{
    int workID = threadIdx.x + (blockDim.x * blockIdx.x);
    
    if (workID < size) 
    {
        x[workID] = 1+ (workID < (size / 2));
    }
}

__global__ void stencil_naive(float* u_new, const float* u_old, 
                               int N, float r) {
    int workID = blockIdx.x * blockDim.x + threadIdx.x;
    
    // your logic here
    // r = K*dt/dx^2
    if (workID > 0 && workID < N-1)
    {
        float ld = u_old[workID+1];
        float rd = u_old[workID-1];
        u_new[workID] = u_old[workID] + r * (ld - 2 * u_old[workID] + rd); 
    }

}

__global__ void stencil_shared(float* u_new, float* u_old,
                                int N, float r) {
    extern __shared__ float tile[];  // size = blockDim.x + 2
    
    int i     = blockIdx.x * blockDim.x + threadIdx.x;
    int s_idx = threadIdx.x + RAD;  // local index with offset for halo
    
    // Step 1: load interior of tile into shared memory
    if (i < N)
    {
        tile[s_idx] = u_old[i]; 
    }
    else
    {
        // If a thread lies on i >= N it has no meaningful information to write
        tile[s_idx] = 0.0f;
    }
    // Step 2: load halo cells (who is responsible for this?)
    if (threadIdx.x < RAD)
    {
         // Left halo
        if (i >= RAD) 
        {
            tile[s_idx - RAD] = u_old[i - RAD]; // Safe read
        } 
        else 
        {
            tile[s_idx - RAD] = tile[s_idx];  // Boundary condition injected safely
        }
        // Right halo
        if (i + blockDim.x < N) 
        {
            tile[s_idx + blockDim.x] = u_old[i + blockDim.x]; // Safe read
        } 
        else 
        {
            tile[s_idx + blockDim.x] = u_old[N-1];  // Boundary condition injected safely
        }
    }
    __syncthreads();

    // Step 3: __syncthreads()
    
    // Step 4: compute stencil using tile[], not u_old[]
    if (i < N)
    {
        u_new[i] = tile[s_idx] + r * (tile[s_idx - 1] - 2.0f * tile[s_idx] + tile[s_idx + 1]); 
    }
}

// One forward-Euler timestep of the 2D heat equation on a grayscale image.
//   u_in   : current image, read-only this step
//   u_out  : next image, written this step
//   kappa  : alpha * dt / h^2  (the lumped diffusion coefficient)
__global__ void heatStep(const float* u_in, float* u_out,
                         int width, int height, float kappa)
{
   // Map this thread to one pixel. col = x, row = y.
   // Consecutive threads in a warp vary in col -> coalesced reads of u_in.
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   if (col >= width || row >= height) return;   // threads off the image do nothing

   // Clamp neighbor coordinates to the edge (Neumann / zero-flux boundary):
   // a missing neighbor is treated as a copy of the edge pixel, so no heat leaks out.
   int left  = max(col - 1, 0);
   int right = min(col + 1, width  - 1);
   int up    = max(row - 1, 0);
   int down  = min(row + 1, height - 1);

   // Flatten (row, col) -> 1D index, row-major:  idx = row * width + col.
   float center  = u_in[row  * width + col];
   float n_left  = u_in[row  * width + left];
   float n_right = u_in[row  * width + right];
   float n_up    = u_in[up   * width + col];
   float n_down  = u_in[down * width + col];

   // Discrete 5-point Laplacian: how much "hotter" the neighbors are than the center.
   float laplacian = n_left + n_right + n_up + n_down - 4.0f * center;

   // Forward Euler: nudge each pixel a little toward the average of its neighbors.
   u_out[row * width + col] = center + kappa * laplacian;
}
/*
void time_march(float* d_u0, float* d_u1, 
                int N, float r, int n_steps) {
    float* src = d_u0;
    float* dst = d_u1;
    
    for (int step = 0; step < n_steps; step++) {
        // launch kernel with src -> dst
        stencil_shared<<<K, TPB, (TPB + (2 * RAD)) * sizeof(float)>>> 
        // swap src and dst
    }
}
*/
int main() 
{
    // Dynamic shared memory size. For 2 ends, allocate needed amount of RAD for Halo Cells. 
    int tileS = (TPB + (2 * RAD)) * sizeof(float);
    float *u_old;
    // Host memory allocating 
    u_old = (float*)malloc(sizeof(float) * ts);
    for (int i = 0; i < ts; i++)
    {
        u_old[i] = 0.0f;
    }
    // setting peak in middle
    u_old[8192] = 1.0f;

    // Device memory allocating
    float *d_u_old, *d_u_new;
    cudaMalloc(&d_u_old, sizeof(float) * ts);
    cudaMalloc(&d_u_new, sizeof(float) * ts);
    cudaMemcpy(d_u_old, u_old, sizeof(float) * ts, cudaMemcpyHostToDevice);

    // Step 1: stencil_naive<<<K, TPB>>>(d_u_new, d_u_old, ts, 0.25);
    
    stencil_shared<<<K, TPB, tileS>>>(d_u_new, d_u_old, ts, 0.25f);
    cudaError_t err = cudaGetLastError();        // catches launch errors
    if (err) printf("launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();               // catches in-kernel errors
    if (err) printf("kernel: %s\n", cudaGetErrorString(err));
    fflush(stdout);        
    cudaMemcpy(u_old, d_u_new, sizeof(float) * ts, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ts; i++)
    {
        if (i == 8191 || i == 8192 || i == 8193)
        {
        printf("GPU Set values %f\n", u_old[i]);
        }
    }
    printf("when t = 0: %f, when f(x,t) = 0: %f\n", u_old[0], u_old[ts -1]);
    free(u_old);
    cudaFree(d_u_old);
    cudaFree(d_u_new);
}   