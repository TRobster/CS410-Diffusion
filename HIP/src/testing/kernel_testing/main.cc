#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <tuple>
#include <functional>
#include "kernel_timing.h"
#include "kernel_correctness.h"
#include "kernels.h"
#include "kernel_utils.h"
#include "../../../../data_pipeline/pgm_filereader.h"

using namespace std;

int block_size_limit =        1024;
int grid_size_limit =      3900000;
int stride_limit =              25;
int overflow_guard =    0x7FFFFFFF;
float tol =                   0.01;

struct kernel_config{
    string version =        "None";
    int   block_size =          -1;
    int    grid_size =          -1;
    int       stride =          -1;
    access_distribution reads = {};
    bool       valid =       false;
};

inline bool validate_stride_config(const kernel_config& config)
{
    return config.stride != -1 || config.stride <= stride_limit;
}

inline bool validate_dimension_config(kernel_config& config, int problem_size)
{
    if (config.block_size < 64 || config.block_size > block_size_limit || config.block_size % 64 != 0) return false;
    if (config.grid_size > grid_size_limit || config.grid_size < -1 || config.grid_size == 0) return false;

    config.grid_size = (config.block_size + problem_size - 1) / config.block_size;
    if (config.grid_size < 1 || config.grid_size > grid_size_limit) return false;

    return true;
}

inline bool validate_adaptive_config(const kernel_config& config)
{
    const access_distribution* read_dist = &(config.reads);
    if (read_dist->smem > 1.0f || read_dist->smem < 0.0f) return false;
    if (read_dist->l1 > 1.0f || read_dist->l1 < 0.0f) return false;
    if (read_dist->l2 > 1.0f || read_dist->l2 < 0.0f) return false;
    if (read_dist->hbm > 1.0f || read_dist->hbm < 0.0f) return false;

    float acc = read_dist->smem + read_dist->l1 + read_dist->l2 + read_dist->hbm;
    return abs(acc - 1.0f) <= tol;
}

void validate_kernel_config(kernel_config& config, int problem_size)
{
    if (config.version == "None") return;
    if (config.version == "stride") config.valid = validate_stride_config(config);
    else if (config.version == "dimension") config.valid = validate_dimension_config(config, problem_size);
    else if (config.version == "adaptive") config.valid = validate_stride_config(config);
}

kernel_config read_config_file(string config_file)
{
    ifstream file(config_file);
    if (!file.is_open()) {
        cerr << "Could not open file: " << config_file << endl;
        return kernel_config{};
    }

    kernel_config config;
    int iter = 0;

    string line;
    while (getline(file, line) && iter < 100) {
        if (line.empty()) continue;

        size_t colon_pos = line.find(':');
        if (colon_pos == string::npos) continue;

        string key = line.substr(0, colon_pos);
        string values = line.substr(colon_pos + 1);
        stringstream ss(values);
        int val;
        float percent_val;
        string version_val;

        if (key == "version") {
            while (ss >> version_val) config.version = version_val;
        } else if (key == "blocksize") {
            while (ss >> val) config.block_size = val;
        } else if (key == "gridsize") {
            while (ss >> val) config.grid_size = val;
        } else if (key == "stride") {
            while (ss >> val) config.stride = val;
        } else if (key == "smem") {
            while (ss >> percent_val) config.reads.smem = percent_val;
        } else if (key == "l1") {
            while (ss >> percent_val) config.reads.l1 = percent_val;
        } else if (key == "l2") {
            while (ss >> percent_val) config.reads.l2 = percent_val;
        } else if (key == "main") {
            while (ss >> percent_val) config.reads.hbm = percent_val;
        }
        
        iter++;
    }

    return config;
}

int main(int argc, char* argv[]) 
{
    if (argc < 3 || argc > 6) {
        std::cout << argv[0] << std::endl;
        cerr << "Usage: " << argv[0]
            << " <synthetic|filename> <timing|correctness> <[kernel config file]> <[rows]> <[cols]>" << endl
            << "for more info on the kernel configs go to the HIP/info/kernel_configs.txt file" << endl;
        return 1;
    }

    string data_source = argv[1];
    string mode        = argv[2];
    string config_file = argv[3];

    int rows = -1, cols = -1;

    if (data_source == "synthetic") {
        if (argc < 6) {
            cerr << "Error: synthetic mode requires <rows> and <cols> arguments" << endl;
            return 1;
        }
        rows = stoi(argv[4]);
        cols = stoi(argv[5]);
    } else {
        auto [r, c] = rows_and_cols(data_source);
        rows = r;
        cols = c;
    }

    if (rows <= 0 || cols <= 0 || (long long)rows * cols > PROBLEM_SIZE_LIMIT) {
        cerr << "Error: invalid or too-large problem size (" << rows << "x" << cols << ")" << endl;
        return 1;
    }

    kernel_config config = read_config_file(config_file);
    validate_kernel_config(config, rows * cols);

    if (!config.valid) {
        cerr << "Error: invalid kernel configuration" << endl;
        return 1;
    }

    int stride         = (config.stride != -1) ? config.stride : 1;
    const float alpha  = 0.1f;
    const int iters    = 100;

    vector<float> u;
    if (data_source == "synthetic") {
        u = generate_u(rows, cols, {0.0f, 1.0f});
    } else {
        u = read_pgm_file(data_source, true);
    }

    if (mode == "timing") {
        time_kernel(u, alpha, rows, cols, stride, iters, config.version,
                    config.block_size, config.grid_size, config.reads);
    } else if (mode == "correctness") {
        vector<float> uniform_field = generate_u(rows, cols, {0.0f, 1.0f}, 1.0f);
        vector<float> u_new(rows * cols, 0.0f);

        bool pass = test_kernel_cpu(u, u_new, vector<float>(rows * cols, 0.0f), rows, cols, alpha, config.version,
                                stride, config.block_size, config.grid_size, config.reads);

        return pass ? 0 : 1;
    } else {
        cerr << "Error: unknown mode '" << mode << "'. Use 'timing' or 'correctness'" << endl;
        return 1;
    }

    return 0;
}