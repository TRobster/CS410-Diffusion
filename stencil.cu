#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>

static int FIXED = 32; // << -- STATIC Int is only accessible on CPU! Not GPU. #FIX 2
__global__ void memTest(int *x, int size) // Pass CPU integer as size. #FIX 2
{
    int workID = threadIdx.x + (blockDim.x * blockIdx.x);
    
    x[workID] = (workID < (size / 2));
}

int main() 
{
    // Host memory allocating 
    int *host;
    host = (int*)malloc(sizeof(int) * FIXED); // << --- MALLOC returns pointer #FIX 1
    // Device memory allocating
    int *dev; 
    cudaMalloc(&dev, sizeof(int) * FIXED); // << --- cudaMALLOC returns error code #FIX 1
    cudaMemcpy(dev, host, sizeof(int) * FIXED, cudaMemcpyHostToDevice);

    memTest<<<1, 32>>>(dev, FIXED);

    cudaMemcpy(host, dev, sizeof(int) * FIXED, cudaMemcpyDeviceToHost);

    for (int i = 0; i < FIXED; i++)
    {
        printf("Binary values %d", host[i]);
    }

}   