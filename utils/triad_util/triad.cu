// triad.cu — STREAM‑style Triad kernel with POSIX signal handling
// Compile: nvcc -O3 -lineinfo -std=c++17 -arch=sm_90a triad.cu -o triad
// Run:     ./triad [elements] [iterations]
//          Ctrl‑C to interrupt gracefully (buffers will be freed).

#include <cuda_runtime.h>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

// ————————————————————————————————
// Global flag toggled by SIGINT / SIGTERM
static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
}

// ————————————————————————————————
// Triad kernel: a[i] = b[i] + alpha * c[i]
__global__ void triad(float* __restrict__ a,
                      const float* __restrict__ b,
                      const float* __restrict__ c,
                      float alpha,
                      size_t n) {
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) {
        a[i] = b[i] + alpha * c[i];
    }
}

// ————————————————————————————————
int main(int argc, char** argv) {
    // 1. Parse CLI: elements & iterations
    const size_t N          = (argc > 1 ? std::stoull(argv[1]) : (1ULL << 28)); // default 256 Mi elements ≈ 1 GiB
    const int    max_iters  = (argc > 2 ? std::atoi(argv[2])   : 100);          // default 100 loops
    const float  alpha      = 3.1415926f;

    // 2. Register signal handlers
    struct sigaction sa {};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // 3. Allocate GPU buffers
    size_t bytes = N * sizeof(float);
    float *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);

    // (Optional) warm‑up write to avoid first‑touch overhead
    cudaMemset(a, 0, bytes);

    // 4. Kernel configuration
    constexpr int  BLOCK = 1024;
    int grid = (N + BLOCK - 1) / BLOCK;
    grid = (grid > 65535 ? 65535 : grid); // limit per‑kernel gridsize

    // 5. Timing helpers
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    double logical_bytes = 3.0 * static_cast<double>(bytes); // 2R + 1W per element

    for (int iter = 0; iter < max_iters && g_running.load(std::memory_order_relaxed); ++iter) {
        cudaEventRecord(t0);
        triad<<<grid, BLOCK>>>(a, b, c, alpha, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        double bw = (logical_bytes / (ms * 1e-3)) / (1ULL << 30); // GB/s

        printf("Iter %3d | %.3f ms | Effective BW %.2f GB/s\n", iter, ms, bw);
    }

    // 6. Cleanup
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    puts("Done. (freed GPU buffers)");
    return 0;
}

