#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

<<<<<<< HEAD
static int FIXED = 32; // << -- STATIC Int is only accessible on CPU! Not GPU. #FIX 2
=======

//static int FIXED = 32; // << -- STATIC Int is only accessible on CPU! Not GPU. #FIX 2
>>>>>>> refs/remotes/origin/main
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
    float *u_new; 
    // Host memory allocating 
    u_old = (float*)malloc(sizeof(float) * 10000);
    for (int i = 0; i < 10000; i++)
    {
        if (i > 0 && i < 10000-1)
        {
            u_old[i] = 1.0;
        }
        else
        {
            u_old[i] = 0.0;
            printf("%f\n", u_old[i]);
        }
    }
    /*
    block 1
    int *host;
    host = (int*)malloc(sizeof(int) * FIXED); // << --- MALLOC returns pointer #FIX 1
    // Device memory allocating
    float *u_new; 
    cudaMalloc(&u_new, sizeof(float) * 10000);
    cudaMemcpy(u_new, u_old, sizeof(float) * 10000, cudaMemcpyHostToDevice);
    */
    /*
    block 2 (not needed for right now)
    int *dev; 
    cudaMalloc(&dev, sizeof(int) * FIXED); // << --- cudaMALLOC returns error code #FIX 1
    cudaMemcpy(dev, host, sizeof(int) * FIXED, cudaMemcpyHostToDevice);

    memTest<<<1, 20>>>(dev, FIXED);

    cudaMemcpy(host, dev, sizeof(int) * FIXED, cudaMemcpyDeviceToHost);

    for (int i = 0; i < FIXED; i++)
    {
        if (host[i] > 0)
        {
        printf("GPU Set values %d\n", host[i]);
        }
        else
        {
        printf("Garbage data %d\n", host[i]);
        }
    }
    */
    free(u_old);
}
