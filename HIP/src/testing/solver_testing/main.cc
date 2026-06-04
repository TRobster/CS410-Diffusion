#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include "solver_timing.h"
#include "solver_correctness.h"
#include "heat_equation.h"
#include "kernel_utils.h"
#include "../../../../data_pipeline/pgm_filereader.h"

using namespace std;

struct solver_config {
    int  gridsize  =   -1;
    int  blocksize =  256;
    int  stride    =    1;
    bool valid     = false;
};

solver_config read_config_file(string config_file)
{
    ifstream file(config_file);
    if (!file.is_open()) {
        cerr << "Could not open file: " << config_file << endl;
        return solver_config{};
    }

    solver_config config;
    config.valid = true;
    int iter = 0;

    string line;
    while (getline(file, line) && iter < 100) {
        if (line.empty()) continue;

        size_t colon_pos = line.find(':');
        if (colon_pos == string::npos) continue;

        string key    = line.substr(0, colon_pos);
        string values = line.substr(colon_pos + 1);
        stringstream ss(values);
        int val;

        if (key == "gridsize") {
            while (ss >> val) config.gridsize = val;
        } else if (key == "blocksize") {
            while (ss >> val) config.blocksize = val;
        } else if (key == "stride") {
            while (ss >> val) config.stride = val;
        }

        iter++;
    }

    return config;
}

int main(int argc, char* argv[])
{
    if (argc < 6 || argc > 8) {
        cerr << "Usage: " << argv[0]
             << " <synthetic|filename> <timing|correctness> <config_file> <alpha> <iters> [rows] [cols]" << endl;
        return 1;
    }

    string data_source = argv[1];
    string mode        = argv[2];
    string config_file = argv[3];
    float  alpha       = stof(argv[4]);
    int    iters       = stoi(argv[5]);

    int rows = -1, cols = -1;

    if (data_source == "synthetic") {
        if (argc < 8) {
            cerr << "Error: synthetic mode requires <rows> and <cols> arguments" << endl;
            return 1;
        }
        rows = stoi(argv[6]);
        cols = stoi(argv[7]);
    } else {
        auto [r, c] = rows_and_cols(data_source);
        rows = r;
        cols = c;
    }

    if (rows <= 0 || cols <= 0 || (long long)rows * cols > PROBLEM_SIZE_LIMIT) {
        cerr << "Error: invalid or too-large problem size (" << rows << "x" << cols << ")" << endl;
        return 1;
    }

    solver_config config = read_config_file(config_file);
    if (!config.valid) {
        cerr << "Error: invalid solver configuration" << endl;
        return 1;
    }

    vector<float> u;
    if (data_source == "synthetic") {
        u = generate_u(rows, cols, {0.0f, 1.0f});
    } else {
        u = read_pgm_file(data_source, true);
    }

    if (mode == "timing") {
        time_solver(u, alpha, rows, cols, iters,
                    config.gridsize, config.blocksize, config.stride);
    } else if (mode == "correctness") {
        vector<float> uniform_field = generate_u(rows, cols, {0.0f, 1.0f}, 1.0f);

        bool pass = true;
        pass &= test_solver_uniform_field(uniform_field, rows, cols, alpha, iters,
                                          config.gridsize, config.blocksize, config.stride);
        pass &= test_solver_cpu(u, rows, cols, alpha, iters,
                                config.gridsize, config.blocksize, config.stride);
        return pass ? 0 : 1;
    } else {
        cerr << "Error: unknown mode '" << mode << "'. Use 'timing' or 'correctness'" << endl;
        return 1;
    }

    return 0;
}
