#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


static int ts = 16384; // << -- STATIC Int is only accessible on CPU! Not GPU. #FIX 2
__global__ void memTest(int *x, int size) // Pass CPU integer as size. #FIX 2
{
    int workID = threadIdx.x + (blockDim.x * blockIdx.x);
    
    if (workID < size) 
    {
        x[workID] = 1+ (workID < (size / 2));
    }
}

__global__ void stencil_naive(float* u_new, float* u_old, 
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
int main() 
{
    float *u_old;
    // Host memory allocating 
    u_old = (float*)malloc(sizeof(float) * ts);
    for (int i = 0; i < ts; i++)
    {
        u_old[i] = 0.0f;
    }
    
    u_old[8192] = 1.0f;
    //int *host;
    //host = (int*)malloc(sizeof(int) * FIXED); // << --- MALLOC returns pointer #FIX 1
    // Device memory allocating
    float *d_u_old, *d_u_new;
    cudaMalloc(&d_u_old, sizeof(float) * ts);
    cudaMalloc(&d_u_new, sizeof(float) * ts);
    cudaMemcpy(d_u_old, u_old, sizeof(float) * ts, cudaMemcpyHostToDevice);
    stencil_naive<<<32, 512>>>(d_u_new, d_u_old, 16384, 0.25f);
    
    /*
    int *dev; 
    cudaMalloc(&dev, sizeof(int) * FIXED); // << --- cudaMALLOC returns error code #FIX 1
    cudaMemcpy(dev, host, sizeof(int) * FIXED, cudaMemcpyHostToDevice);

    memTest<<<1, 20>>>(dev, FIXED);
    */
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