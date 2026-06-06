#ifndef UTILS_H
#define UTILS_H

#include <hip/hip_runtime.h>
#include <utility>
#include <vector>

// Error-checking macro for HIP API calls
#define HIP_CHECK(cmd) do { hipError_t e = (cmd); if (e != hipSuccess) { \
    fprintf(stderr,"HIP ERROR %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

//Limit to data you can load onto the GPA
inline int PROBLEM_SIZE_LIMIT = 250000000;
inline int L1_LATENCY = 4;
inline int L2_LATENCY = 40;
inline int SMEM_LATENCY = 4;

//Load percentage struct
struct access_distribution
{
    float smem;
    float l1;
    float l2;
    float hbm;
};

//===================================
// GPU information function
//===================================

inline size_t get_max_smem(); 
inline int get_smem_latency();
inline int get_smem_bandwidth();
inline int get_global_latency();
inline int get_global_mem_bandwidth();
inline int get_l1_bandwidth();
inline int get_compute_units();
inline int get_total_simd_units(int deviceId = 0);
inline double get_gpu_memory_capacity();
inline void print_amd_gpu_model();
int select_device();

//===================================
// Dimension Helpers
//===================================
std::pair<int, int> determine_block_dimensions(int blocksize);
int determine_avg_latency(access_distribution accesses);
int determine_avg_bandwidth(access_distribution accesses);
int determine_ideal_cu_occupancy(access_distribution accesses);
int determine_stride(int total_simd, int rank, uint64_t nnz);
std::pair<int,int> determine_dimensions_stride(int arr_size, int block_size, int stride);
std::pair<int, int> determine_dimensions_occupancy(int arr_size, int block_size, int occupancy);
std::pair<std::pair<int, int>, int> determine_dimensions_and_stride_occupancy(int arr_size, int block_size, int occupancy);

#endif
