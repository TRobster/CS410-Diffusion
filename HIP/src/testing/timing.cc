#include "timing.h"
#include "kernels.h"
#include "kernel_utils.h"
#include <cmath>
#include <numeric>

void clean_timing_data(
    std::vector<float> times,
    int                sd_tolerance
) 
{
    if (times.empty()) return;

    float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

    float variance = 0.0f;
    for (float t : times) variance += (t - mean) * (t - mean);
    variance /= times.size();
    float sd = std::sqrt(variance);

    std::vector<float> cleaned;
    for (float t : times) {
        if (std::fabs(t - mean) <= sd_tolerance * sd)
            cleaned.push_back(t);
    }
    if (cleaned.empty()) cleaned = times;

    float clean_mean = std::accumulate(cleaned.begin(), cleaned.end(), 0.0f) / cleaned.size();
    float clean_min  = *std::min_element(cleaned.begin(), cleaned.end());
    float clean_max  = *std::max_element(cleaned.begin(), cleaned.end());

    std::cout << "Timing (" << cleaned.size() << "/" << times.size() << " samples after "
              << sd_tolerance << "-SD filter):\n"
              << "  Mean: " << clean_mean << " ms\n"
              << "  Min:  " << clean_min  << " ms\n"
              << "  Max:  " << clean_max  << " ms\n";
}

void time_kernel(
    std::vector<float> u,
    float              alpha,
    int                rows,
    int                cols,
    int                kernel_stride,
    int                iterations,
    std::string        kernel_version
) 
{
    int deviceId = select_device();
    size_t bytes = rows * cols * sizeof(float);

    float* d_u     = nullptr;
    float* d_u_new = nullptr;
    HIP_CHECK(hipMalloc(&d_u,     bytes));
    HIP_CHECK(hipMalloc(&d_u_new, bytes));
    HIP_CHECK(hipMemcpy(d_u,     u.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u_new, u.data(), bytes, hipMemcpyHostToDevice));

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    std::vector<float> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        HIP_CHECK(hipEventRecord(ev_start));

        // "default" is the only kernel version currently
        heat_equation_neumann_step(d_u, d_u_new, rows, cols, alpha, kernel_stride, deviceId);

        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        times.push_back(ms);

        std::swap(d_u, d_u_new);
    }

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_u_new));

    clean_timing_data(times, 2);
}
