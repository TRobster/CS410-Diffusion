#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>

static int FIXED = 32;
__global__ void memTest(int *x)
{
    int workID = threadIdx.x + (blockDim.x * blockIdx.x);
    
    x[workID] = (workID < (FIXED / 2));
}

int main() 
{
    // Host memory allocating 
    int *host;
    host = malloc(host, sizeof(int) * FIXED);
    // Device memory allocating
    int *dev; 
    dev = cudaMalloc(&dev, sizeof(int) * FIXED);
    cudaMemcpy(dev, host, sizeof(int) * FIXED, cudaMemcpyHostToDevice);

    memTest<<<1, 32>>>(dev);

    cudaMemcpy(host, dev, sizeof(int) * FIXED, cudaMemcpyDeviceToHost);

    for (int i = 0; i < FIXED; i++)
    {
        printf("Binary values %d", host[i]);
    }

}   