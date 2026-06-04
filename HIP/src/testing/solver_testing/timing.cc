#include "solver_timing.h"
#include "heat_equation.h"
#include <cmath>
#include <numeric>
#include <chrono>

void clean_timing_data(
    std::vector<float> times,
    int                sd_tolerance
) {
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

void time_solver(
    std::vector<float> u,
    float              alpha,
    int                rows,
    int                cols,
    int                iters,
    int                gridsize,
    int                blocksize,
    int                stride
) {
    // warmup — avoids measuring JIT / driver init costs
    for(int i = 0; i < 5; i++) heat_equation_2d(u, rows, cols, alpha, iters, gridsize, blocksize, stride);

    const int repetitions = 10;
    std::vector<float> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; r++) {
        auto t0  = std::chrono::high_resolution_clock::now();
        heat_equation_2d(u, rows, cols, alpha, iters, gridsize, blocksize, stride);
        auto t1  = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        std::cout << ms << std::endl;
        times.push_back(ms);
    }

    clean_timing_data(times, 2);
}
