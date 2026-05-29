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
    tile[s_idx] = u_old[i]; 
    // Step 2: load halo cells (who is responsible for this?)
    if (threadIdx.x < RAD)
    {
        tile[s_idx - RAD] = u_old[i - RAD]; 
        tile[s_idx + blockDim.x] = u_old[i + blockDim.x];
    }
    __syncthreads();

    // Step 3: __syncthreads()
    
    // Step 4: compute stencil using tile[], not u_old[]
    u_new[i] = tile[s_idx] + r * (tile[s_idx - 1] - 2.0f * tile[s_idx] + tile[s_idx + 1]); 
}

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
    
    stencil_shared<<<K, TPB, tileS>>>(d_u_new, d_u_old, ts, 0.25);
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