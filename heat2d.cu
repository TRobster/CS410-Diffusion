// heat2d.cu
//
// 2D heat-equation image blur on the GPU, written for clarity over speed.
//
// The idea: treat each pixel's brightness as "temperature" and let it diffuse.
// One forward-Euler timestep nudges every pixel toward the average of its four
// neighbors (a 5-point Laplacian). Repeat many steps and the image blurs --
// equivalent to convolving with a Gaussian whose width grows with the number
// of steps.
//
// This version reads neighbors straight from global memory (no shared-memory
// tiling). That keeps the kernel a direct translation of the math; tiling is
// an optimization you can layer on later without changing what is computed.
//
// Build:   nvcc heat2d.cu -o heat2d
// Run:     ./heat2d
// Output:  before.pgm and after.pgm  (grayscale images you can compare)
//
// Viewing PGM on a headless cluster: either scp the files to your laptop, or
// convert them on the node, e.g.  `convert after.pgm after.png` (ImageMagick).

#include <cstdio>
#include <cstdlib>
#include <utility>          // std::swap
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Tiny error-check macro. Wrap every CUDA call so failures point at a line
// number instead of silently producing garbage.
// ---------------------------------------------------------------------------
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s (at %s:%d)\n",                   \
                    cudaGetErrorString(_e), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// One forward-Euler timestep of the 2D heat equation on a grayscale image.
//   u_in   : current image, read-only this step
//   u_out  : next image, written this step
//   kappa  : alpha * dt / h^2  (the lumped diffusion coefficient)
// ---------------------------------------------------------------------------
__global__ void heatStep(const float* u_in, float* u_out,
                          int width, int height, float kappa)
{
    // Map this thread to one pixel. col = x, row = y.
    // Consecutive threads in a warp vary in col -> coalesced reads of u_in.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;   // threads off the image do nothing

    // Clamp neighbor coordinates to the edge (Neumann / zero-flux boundary):
    // a missing neighbor is treated as a copy of the edge pixel, so no heat
    // leaks out of the image and the border does not darken.
    int left  = max(col - 1, 0);
    int right = min(col + 1, width  - 1);
    int up    = max(row - 1, 0);
    int down  = min(row + 1, height - 1);

    // Flatten (row, col) -> 1D index, row-major:  idx = row * width + col.
    float center  = u_in[row  * width + col];
    float n_left  = u_in[row  * width + left];
    float n_right = u_in[row  * width + right];
    float n_up    = u_in[up   * width + col];
    float n_down  = u_in[down * width + col];

    // Discrete 5-point Laplacian: how much hotter the neighbors are than center.
    float laplacian = n_left + n_right + n_up + n_down - 4.0f * center;

    // Forward Euler: step a little toward the neighborhood average.
    u_out[row * width + col] = center + kappa * laplacian;
}

// ---------------------------------------------------------------------------
// Build a synthetic test image: black background, a bright central square,
// and a few isolated bright points. The square's sharp edges visibly soften
// and the points spread into blobs as diffusion runs.
// ---------------------------------------------------------------------------
static void makeTestImage(float* img, int width, int height)
{
    for (int i = 0; i < width * height; ++i) img[i] = 0.0f;   // black

    // Centered bright square.
    for (int row = height / 4; row < 3 * height / 4; ++row)
        for (int col = width / 4; col < 3 * width / 4; ++col)
            img[row * width + col] = 255.0f;

    // A few single-pixel bright points: {row, col}.
    const int pts[][2] = { {32, 32}, {40, 220}, {220, 40}, {210, 210} };
    for (const auto& p : pts)
        img[p[0] * width + p[1]] = 255.0f;
}


// Read a binary PGM (P5) into a freshly malloc'd float array.
// Sets *width and *height from the file header. Minimal parser: assumes "P5",
// maxval 255, no comment lines. (Matches what your writePGM and ImageMagick emit.)
static float* readPGM(const char* filename, int* width, int* height)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", filename); exit(EXIT_FAILURE); }

    char magic[3] = {0};
    int maxval = 0;
    if (fscanf(f, "%2s %d %d %d", magic, width, height, &maxval) != 4) {
        fprintf(stderr, "bad PGM header in %s\n", filename); exit(EXIT_FAILURE);
    }
    if (magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "%s is not a binary (P5) PGM\n", filename); exit(EXIT_FAILURE);
    }
    fgetc(f);   // consume the single whitespace byte between header and pixel data

    int n = (*width) * (*height);
    float* img = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i)
        img[i] = (float)fgetc(f);   // bytes 0..255 -> float, your existing value range
    fclose(f);
    return img;
}
// ---------------------------------------------------------------------------
// Write a grayscale image as a binary PGM (P5) file. Values are clamped to
// [0, 255] and rounded to bytes.
// ---------------------------------------------------------------------------
static void writePGM(const char* filename, const float* img, int width, int height)
{
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; ++i) {
        float v = img[i];
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        unsigned char byte = (unsigned char)(v + 0.5f);   // round to nearest
        fwrite(&byte, 1, 1, f);
    }
    fclose(f);
}



int main(void)
{

    const float kappa    = 0.10f;   // diffusion coefficient; must be <= 0.25 for stability
    const int   numSteps = 200;     // more steps = more blur (sigma grows ~ sqrt(steps))
    int width, height; 
    const size_t numPixels = (size_t)width * height;
    const size_t numBytes  = numPixels * sizeof(float);

    // --- Host image ---
    float* h_img  = readPGM("baboon.pgm", &width, &height);
    if (!h_img) { fprintf(stderr, "host malloc failed\n"); return EXIT_FAILURE; }
    makeTestImage(h_img, width, height);
    writePGM("before.pgm", h_img, width, height);

    // --- Device buffers (ping-pong: read one, write the other, then swap) ---
    float *d_curr = nullptr, *d_next = nullptr;
    CHECK(cudaMalloc(&d_curr, numBytes));
    CHECK(cudaMalloc(&d_next, numBytes));
    CHECK(cudaMemcpy(d_curr, h_img, numBytes, cudaMemcpyHostToDevice));

    // Launch configuration: one thread per pixel
    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Timing setup before step
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Optional warm-up: the very first launch pays one-time costs (context setup,
    // module load). Running one throwaway step keeps that out of the measurement.
    heatStep<<<grid, block>>>(d_curr, d_next, width, height, kappa);
    cudaDeviceSynchronize();
    // --- Time stepping ---
    cudaEventRecord(start);
    for (int step = 0; step < numSteps; ++step) {
        heatStep<<<grid, block>>>(d_curr, d_next, width, height, kappa);
        CHECK(cudaGetLastError());     // catch bad launch configs etc.
        std::swap(d_curr, d_next);     // the freshly written buffer becomes current
    }
    CHECK(cudaDeviceSynchronize());    // wait for the GPU to finish all steps
    cudaError_t err = cudaGetLastError();        // catches launch errors
    if (err) printf("launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();               // catches in-kernel errors
    if (err) printf("kernel: %s\n", cudaGetErrorString(err));
    fflush(stdout);        

    cudaEventRecord(stop);       
    cudaEventSynchronize(stop);          // wait until the stop event truly completes
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%d steps: %.3f ms total, %.4f ms/step\n", numSteps, ms, ms / numSteps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // After the loop d_curr points at the most recently written image.
    CHECK(cudaMemcpy(h_img, d_curr, numBytes, cudaMemcpyDeviceToHost));
    writePGM("after.pgm", h_img, width, height);

    printf("Done. Wrote before.pgm and after.pgm (%d steps, kappa = %.3f).\n",
           numSteps, kappa);

    // --- Cleanup ---
    CHECK(cudaFree(d_curr));
    CHECK(cudaFree(d_next));
    free(h_img);
    return 0;
}
