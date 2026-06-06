#include <hip/hip_runtime.h>
#include <utility>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include "kernel_utils.h"

using namespace std;

//Get maximum shared memory
inline size_t get_max_smem()
{
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // The key property for kernel launch limitation:
    return props.sharedMemPerBlock;
}

// Shared memory latency in cycles (typical AMD GCN/CDNA value)
inline int get_smem_latency()
{
    return 20;
}

// Shared memory bandwidth in GB/s estimated from CU count and clock rate
inline int get_smem_bandwidth()
{
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // Each CU can read 128 bytes/cycle from shared memory; clockRate is in kHz
    double clock_ghz = props.clockRate / 1e6;
    return (int)(props.multiProcessorCount * 128.0 * clock_ghz);
}

// Global (HBM) memory latency in cycles estimated from 300 ns @ device clock
inline int get_global_latency()
{
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // 300 ns * clockRate (Hz) gives cycles; clockRate is in kHz
    double latency_cycles = 300e-9 * (props.clockRate * 1e3);
    return (int)latency_cycles;
}

// Theoretical global memory bandwidth in GB/s from device properties
inline int get_global_mem_bandwidth()
{
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // bandwidth (GB/s) = memClockRate(kHz) * busWidth(bits) * 2(DDR) / 8(bits->bytes) / 1e6
    double bw = (double)props.memoryClockRate * props.memoryBusWidth * 2.0 / 8.0 / 1e6;
    return (int)bw;
}

// L1 cache bandwidth in GB/s estimated from CU count and clock rate
inline int get_l1_bandwidth()
{
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // Each CU can read 64 bytes/cycle from L1; clockRate is in kHz
    double clock_ghz = props.clockRate / 1e6;
    return (int)(props.multiProcessorCount * 64.0 * clock_ghz);
}

//Get the number of compute units on the GPU
inline int get_compute_units() {
    int deviceId = 0; // Use the first GPU
    hipDeviceProp_t props;

    // Query device properties
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    return props.multiProcessorCount;
}

//Total number of SIMD units on the GPU
inline int get_total_simd_units(int deviceId) {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    int cu_count = props.multiProcessorCount;
    int simds_per_cu = 4; // Default for CDNA/GCN (4 SIMD16 per CU)

    string arch(props.gcnArchName);

    // RDNA architectures (gfx10xx for RDNA1/2, gfx11xx for RDNA3)
    if (arch.find("gfx10") != string::npos || arch.find("gfx11") != string::npos) {
        simds_per_cu = 2; // RDNA architectures typically have 2 SIMD32s per CU
    } else if (arch.find("gfx9") != string::npos) {
        simds_per_cu = 4; // CDNA/Vega/GCN architectures have 4 SIMD16s per CU
    }

    return cu_count * simds_per_cu;
}

//Get the memory capacity of the GPU
double get_gpu_memory_capacity() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cout << "No HIP-compatible devices found.\n";
        return -1.0;
    }

    // Assuming we check device 0
    int device = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    // props.totalGlobalMem gives the memory size in bytes
    unsigned long long total_bytes = props.totalGlobalMem;

    // Convert bytes to GB for readability
    double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

    // You can also check available memory at runtime
    size_t free, total;
    HIP_CHECK(hipMemGetInfo(&free, &total));
    double free_gb = static_cast<double>(free) / (1024.0 * 1024.0 * 1024.0);
    return free_gb;
}

//Prints information about GPU
void print_amd_gpu_model() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cout << "No HIP-compatible AMD devices found.\n";
        return;
    }

    // Loop through all found devices
    for (int i = 0; i < 1; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));

        // The name of the GPU is stored in props.name
        cout << "Device: " << props.name << "\n";
        cout << "Arch: " << props.gcnArchName << "\n";
        cout << "Shared Memory: " << props.sharedMemPerBlock << "bytes\n";
        cout << "Currently Free VRAM: " << get_gpu_memory_capacity() << " GB\n";
    }
}

// Returns the ID of the GPU with the most free VRAM and sets it as the active device.
int select_device() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "No HIP-compatible devices found." << endl;
        exit(EXIT_FAILURE);
    }

    int best_device = 0;
    size_t max_free = 0;

    for (int i = 0; i < deviceCount; i++) {
        HIP_CHECK(hipSetDevice(i));
        size_t free_mem, total_mem;
        HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
        if (free_mem > max_free) {
            max_free = free_mem;
            best_device = i;
        }
    }

    HIP_CHECK(hipSetDevice(best_device));
    return best_device;
}

//Determine dimensions of block based on the block size
std::pair<int, int> determine_block_dimensions(int blocksize)
{
    int y = (int)std::sqrt((double)blocksize);
    while (y > 1 && blocksize % y != 0)
        y--;
    return {blocksize / y, y};
}

// Average of shared memory, global, and L1 latencies (in cycles)
int determine_avg_latency(access_distribution accesses) {

    float latency = accesses.hbm * get_global_latency() + accesses.smem * SMEM_LATENCY
    + accesses.l1 * L1_LATENCY + accesses.l2 * L2_LATENCY;
    return static_cast<int>(latency);
}

// Average of shared memory, global, and L1 bandwidths (in GB/s)
int determine_avg_bandwidth(access_distribution accesses) {
    return accesses.smem * get_smem_bandwidth() + accesses.l1 * get_l1_bandwidth() 
    + accesses.l2 * get_l1_bandwidth() + accesses.hbm * get_global_mem_bandwidth();
}

// Minimum wavefronts per CU needed to hide average memory latency (Little's Law)
int determine_ideal_cu_occupancy(access_distribution accesses) {
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // Hardware max wavefronts per CU differs by architecture
    string arch(props.gcnArchName);
    int max_wavefronts_per_cu;
    if (arch.find("gfx10") != string::npos || arch.find("gfx11") != string::npos) {
        max_wavefronts_per_cu = 32; // RDNA: 2 SIMDs x 16 wavefronts
    } else {
        max_wavefronts_per_cu = 40; // GCN/CDNA: 4 SIMDs x 10 wavefronts
    }

    int latency = determine_avg_latency(accesses);
    const int issue_cycles = 4; // AMD GCN/CDNA: one instruction issued every 4 cycles
    int wavefronts = (latency + issue_cycles - 1) / issue_cycles;
    return min(wavefronts, max_wavefronts_per_cu);
}

// Stride sized so each SIMD unit processes an equal share of the NNZ elements,
int determine_stride(int total_simd, int rank, uint64_t nnz) {
    if (total_simd <= 0 || rank <= 0) return 1;
    uint64_t work_per_simd = (nnz + (uint64_t)total_simd - 1) / (uint64_t)total_simd;
    uint64_t stride = work_per_simd / (uint64_t)rank;
    return (int)std::max((uint64_t)1, stride);
}

// Grid dimensions when each thread steps through the array with the given stride
pair<int,int> determine_dimensions_stride(int arr_size, int block_size, int stride) {
    int threads_needed = (arr_size + stride - 1) / stride;
    int gridsize = (threads_needed + block_size - 1) / block_size;
    return {gridsize, block_size};
}

// Grid dimensions that achieve the target wavefront occupancy across all CUs,
// while still covering every element of the array
pair<pair<int, int>, int> determine_dimensions_and_stride_occupancy(int arr_size, int block_size, int occupancy) {

    const int wavefront_size = 64; // AMD wavefront is 64 threads
    int cu_count = get_compute_units();
    int total_threads = occupancy * wavefront_size * cu_count;
    int gridsize = max(1, total_threads / block_size);
    // Ensure full coverage of the array
    int min_gridsize = ((arr_size / 25) + block_size - 1) / block_size;
    gridsize = max(gridsize, min_gridsize);
    int total_threads_launched = block_size * gridsize;
    int stride = max(1, (total_threads_launched + arr_size - 1) / total_threads_launched);
    return {{gridsize, block_size}, stride};
}
