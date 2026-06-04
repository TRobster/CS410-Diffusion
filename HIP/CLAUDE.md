# HIP 2D Heat Equation — Project Overview

This subdirectory implements a GPU-accelerated 2D heat equation solver using AMD HIP. The solver applies an explicit Euler finite-difference scheme with Neumann (zero-flux) boundary conditions over a 2D grid, and supports multiple kernel tuning strategies.

## Directory Structure

```
HIP/
├── include/               # Shared headers (included by all translation units)
│   ├── heat_equation.h    # Top-level solver API: heat_equation_2d()
│   ├── kernels.h          # Three kernel launcher declarations + BLOCK_X/BLOCK_Y defines
│   ├── kernel_utils.h     # GPU info helpers, dimension math, access_distribution struct
│   ├── correctness.h      # CPU reference solver and test harness declarations
│   └── timing.h           # Timing utilities
│
├── src/
│   ├── implementation/
│   │   ├── kernels/       # Builds libheat.a (the GPU kernel library)
│   │   │   ├── Makefile
│   │   │   ├── kernels.cc         # __global__ kernel + three launcher implementations
│   │   │   ├── kernel_utils.cc    # GPU info queries, dimension/stride heuristics
│   │   │   └── obj/               # Compiled object files (generated)
│   │   └── 2d_heat_equation/      # Top-level solver wrapper (heat_equation_2d)
│   │       ├── Makefile
│   │       └── heat_equation.cc   # Orchestrates memory alloc, kernel dispatch, copy-back
│   │
│   └── testing/           # Test/benchmark harness — builds heat_test binary
│       ├── Makefile
│       ├── main.cc            # Entry point; parses config files and dispatches timing/correctness
│       ├── correctness.cc     # CPU reference solver + comparison tests
│       ├── timing.cc          # Kernel timing utilities
│       ├── config_files/      # Kernel configuration files (text, colon-separated key:value)
│       │   ├── basic.txt      # version: default (no tuning)
│       │   └── stride.txt     # version: stride, stride: 8
│       └── profiling/
│           ├── profile.sh             # Runs rocprofv3 then extracts counters to CSV
│           ├── yaml_files/            # rocprofv3 counter-collection configs
│           │   ├── basic.yaml
│           │   └── memory.yaml
│           └── csv_cleanup/
│               └── extract_counters.py  # Post-processes profiler CSVs into one output.csv
│
├── info/
│   ├── kernel_configs.txt      # Documents config file format for all three kernel versions
│   └── architecture_flags.txt  # AMD GPU arch codes for --offload-arch= (RDNA, CDNA, GCN)
│
└── performance_data/
    └── output.csv              # Accumulated profiling results
```

## Build System

Two independent Makefiles that must be built in order:

1. **`src/implementation/kernels/Makefile`** — compiles `kernels.cc` and `kernel_utils.cc` into `libheat.a`
   - Pass `ARCH=<gfx_code>` to target a specific GPU (e.g. `make ARCH=gfx90a`)
   - Run `rocminfo | grep -E "(Agent |Name: |Marketing Name)"` to find your GPU's code
2. **`src/testing/Makefile`** — links against `libheat.a` (and `libpgm.a` from `data_pipeline/`) to produce `heat_test`
   - Top-level `make all` in `src/testing/` triggers both sub-builds automatically

## Running the Binary

```
./heat_test <synthetic|<pgm_file>> <timing|correctness> <config_file> [rows] [cols]
```

- `synthetic` requires explicit `rows` and `cols`; a PGM filename reads dimensions from the file
- `config_file` is a path to a key:value text file (see below)

## Kernel Versions and Config Files

Config files live in `src/testing/config_files/` and are colon-separated `key: value` pairs.

| Version | Config keys | What it tunes |
|---|---|---|
| `stride` | `stride: <int>` | Tiles-per-block; higher stride = fewer blocks, more L2 reuse |
| `dimension` | `blocksize: <int>`, `gridsize: <int>` | Explicit block and grid sizes |
| `adaptive` | `smem/l1/l2/main: <float>` (must sum to ~1.0) | Infers occupancy from estimated memory hierarchy access fractions |

The `default` version (used in `basic.txt`) runs with stride=1 and no tuning.

## GPU Kernel Details

Single `__global__` kernel (`heat_equation_neumann_kernel` in `kernels.cc`) covers all three launch modes:
- Fixed block shape: `BLOCK_X=16`, `BLOCK_Y=16` (256 threads/block)
- Loads a `(BLOCK_Y+2) × (BLOCK_X+2)` shared-memory tile per block with a 1-cell halo
- Neumann BCs via index clamping (boundary cells mirror themselves as out-of-bounds neighbors)
- Stride > 1 makes each block process multiple tiles in sequence; consecutive tiles share warm L2 data
- Stability requires `alpha = kappa * dt ≤ 0.25`

## Profiling

From `src/testing/profiling/`:
```
./profile.sh yaml_files/basic.yaml synthetic timing config_files/stride.txt 10000 20000
```
Runs `rocprofv3`, collects hardware counters, then calls `extract_counters.py` to merge pass CSVs into a single `output.csv`. Raw rocprofv3 directories are cleaned up automatically.
