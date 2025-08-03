/*
 * Simple NCCL Send/Recv Latency Test
 * 
 * This program measures P50/P99 latency for NCCL Send/Recv between two ranks
 * using cudaStreamQuery for accurate timing.
 * 
 * Compile: nvcc -o simple_sendrecv_latency simple_sendrecv_latency.cu -lnccl
 * Run: mpirun -np 2 ./simple_sendrecv_latency
 *      or manually on two processes with RANK/NRANKS env vars
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unistd.h>
#include <mpi.h>

// Error checking macros
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(1);                                        \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed: NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(1);                                        \
  }                                                 \
} while(0)

// High-precision timer using std::chrono
class Timer {
  using clock = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock> start_time;
  
public:
  Timer() : start_time(clock::now()) {}
  
  void reset() {
    start_time = clock::now();
  }
  
  double elapsed_us() const {
    auto end_time = clock::now();
    return std::chrono::duration<double, std::micro>(end_time - start_time).count();
  }
};

// Calculate percentiles from sorted vector
double calculate_percentile(const std::vector<double>& sorted_data, double percentile) {
  if (sorted_data.empty()) return 0.0;
  
  double index = (percentile / 100.0) * (sorted_data.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(index));
  size_t upper = static_cast<size_t>(std::ceil(index));
  
  if (lower == upper) {
    return sorted_data[lower];
  }
  
  double weight = index - lower;
  return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight;
}

// Print latency statistics
void print_stats(const std::vector<double>& latencies, size_t bytes) {
  if (latencies.empty()) return;
  
  // Make a copy and sort for percentile calculation
  std::vector<double> sorted = latencies;
  std::sort(sorted.begin(), sorted.end());
  
  // Calculate statistics
  double min_lat = sorted.front();
  double max_lat = sorted.back();
  double p50 = calculate_percentile(sorted, 50.0);
  double p90 = calculate_percentile(sorted, 90.0);
  double p95 = calculate_percentile(sorted, 95.0);
  double p99 = calculate_percentile(sorted, 99.0);
  double p999 = calculate_percentile(sorted, 99.9);
  
  // Calculate average
  double sum = 0.0;
  for (double lat : latencies) sum += lat;
  double avg = sum / latencies.size();
  
  // Calculate standard deviation
  double variance = 0.0;
  for (double lat : latencies) {
    double diff = lat - avg;
    variance += diff * diff;
  }
  double stddev = std::sqrt(variance / latencies.size());
  
  // Print results
  printf("\n========== Latency Statistics (bytes=%zu) ==========\n", bytes);
  printf("Samples: %zu\n", latencies.size());
  printf("Min:     %8.2f us\n", min_lat);
  printf("P50:     %8.2f us\n", p50);
  printf("P90:     %8.2f us\n", p90);
  printf("P95:     %8.2f us\n", p95);
  printf("P99:     %8.2f us\n", p99);
  printf("P99.9:   %8.2f us\n", p999);
  printf("Max:     %8.2f us\n", max_lat);
  printf("Average: %8.2f us (±%.2f us)\n", avg, stddev);
  printf("==================================================\n");
}

int main(int argc, char* argv[]) {
  // Parse command line arguments
  
  int warmup_iters = 100;
  int measure_iters = 1000;
  size_t min_bytes = 8;
  size_t max_bytes = 1024 * 1024; // 1MB
  
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-w") == 0 && i+1 < argc) {
      warmup_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
      measure_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-b") == 0 && i+1 < argc) {
      min_bytes = atoll(argv[++i]);
    } else if (strcmp(argv[i], "-e") == 0 && i+1 < argc) {
      max_bytes = atoll(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [-w warmup] [-n iters] [-b min_bytes] [-e max_bytes]\n", argv[0]);
      printf("  -w: Warmup iterations (default: 100)\n");
      printf("  -n: Measurement iterations (default: 1000)\n");
      printf("  -b: Min message size (default: 8)\n");
      printf("  -e: Max message size (default: 1MB)\n");
      return 0;
    }
  }
  
  // Get rank information
  int rank = 0;
  int nranks = 1;
  
  // Try to get from environment variables (for non-MPI runs)
  char* rank_str = getenv("RANK");
  char* nranks_str = getenv("NRANKS");
  if (rank_str && nranks_str) {
    rank = atoi(rank_str);
    nranks = atoi(nranks_str);
  }
 
  printf("Pass0.1"); 

  #ifdef MPI_SUPPORT
  // If compiled with MPI support, use MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  #endif
  
  if (nranks != 2) {
    if (rank == 0) {
      printf("Error: This test requires exactly 2 ranks (got %d)\n", nranks);
      printf("Run with: mpirun -np 2 %s\n", argv[0]);
      printf("Or: RANK=0 NRANKS=2 ./%s & RANK=1 NRANKS=2 ./%s\n", argv[0], argv[0]);
    }
    return 1;
  }
  
  // Set CUDA device
  int num_devices;
  CUDACHECK(cudaGetDeviceCount(&num_devices));
  int device = rank % num_devices;
  CUDACHECK(cudaSetDevice(device));
  
  // Get device properties
  cudaDeviceProp prop;
  CUDACHECK(cudaGetDeviceProperties(&prop, device));
  
  printf("Rank %d using GPU %d: %s\n", rank, device, prop.name);
  
  // Initialize NCCL
  ncclUniqueId id;
  ncclComm_t comm;
  
  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    // Share ID with rank 1 (in production, use MPI_Bcast or write to file)
    #ifdef MPI_SUPPORT
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    #else
    // Simple file-based sharing for non-MPI mode
    FILE* fp = fopen("/tmp/nccl_id_simple.dat", "wb");
    fwrite(&id, sizeof(id), 1, fp);
    fclose(fp);
    #endif
  } else {
    #ifdef MPI_SUPPORT
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    #else
    // Wait for rank 0 to write the file
    usleep(100000); // 100ms
    FILE* fp = fopen("/tmp/nccl_id_simple.dat", "rb");
    if (!fp) {
      printf("Error: Could not read NCCL ID file\n");
      return 1;
    }
    fread(&id, sizeof(id), 1, fp);
    fclose(fp);
    #endif
  }
  
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  
  // Create CUDA stream
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
  // Print test configuration (rank 0 only)
  if (rank == 0) {
    printf("\n===== Simple NCCL Send/Recv Latency Test =====\n");
    printf("Warmup iterations: %d\n", warmup_iters);
    printf("Measurement iterations: %d\n", measure_iters);
    printf("Message size range: %zu - %zu bytes\n", min_bytes, max_bytes);
    printf("==============================================\n");
  }
  
  // Synchronize before starting
  #ifdef MPI_SUPPORT
  MPI_Barrier(MPI_COMM_WORLD);
  #else
  // Simple sync using NCCL AllReduce
  float sync_val = 1.0f;
  NCCLCHECK(ncclAllReduce(&sync_val, &sync_val, 1, ncclFloat, ncclSum, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  #endif
  
  // Test different message sizes
  for (size_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
    // Allocate buffers
    void* sendbuff;
    void* recvbuff;
    CUDACHECK(cudaMalloc(&sendbuff, bytes));
    CUDACHECK(cudaMalloc(&recvbuff, bytes));
    
    // Initialize buffers
    CUDACHECK(cudaMemset(sendbuff, rank + 1, bytes));
    CUDACHECK(cudaMemset(recvbuff, 0, bytes));
    
    // Calculate element count for NCCL (assuming float for simplicity)
    size_t count = bytes / sizeof(float);
    if (count == 0) count = 1; // At least 1 element
    
    // Warmup iterations
    for (int i = 0; i < warmup_iters; i++) {
      if (rank == 0) {
        NCCLCHECK(ncclSend(sendbuff, count, ncclFloat, 1, comm, stream));
      } else {
        NCCLCHECK(ncclRecv(recvbuff, count, ncclFloat, 0, comm, stream));
      }
      CUDACHECK(cudaStreamSynchronize(stream));
    }
    
    // Measurement iterations
    std::vector<double> latencies;
    latencies.reserve(measure_iters);
    
    for (int iter = 0; iter < measure_iters; iter++) {
      Timer timer;
      
      // Launch NCCL operation
      if (rank == 0) {
        // Rank 0: Send to rank 1
        NCCLCHECK(ncclSend(sendbuff, count, ncclFloat, 1, comm, stream));
      } else {
        // Rank 1: Receive from rank 0
        NCCLCHECK(ncclRecv(recvbuff, count, ncclFloat, 0, comm, stream));
      }
      
      // Poll for completion using cudaStreamQuery
      cudaError_t err = cudaErrorNotReady;
      while (err == cudaErrorNotReady) {
        err = cudaStreamQuery(stream);
      }
      CUDACHECK(err); // Check for any errors
      
      // Record latency
      double latency_us = timer.elapsed_us();
      latencies.push_back(latency_us);
      
      // Small delay between iterations to avoid overloading
      if (iter < measure_iters - 1) {
        usleep(100); // 100 microseconds
      }
    }
    
    // Print statistics (rank 0 only)
    if (rank == 0) {
      print_stats(latencies, bytes);
    }
    
    // Cleanup
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    
    // Sync before next size
    #ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    float sync_val = 1.0f;
    NCCLCHECK(ncclAllReduce(&sync_val, &sync_val, 1, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    #endif
  }
  
  // Cleanup
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));
  
  #ifdef MPI_SUPPORT
  MPI_Finalize();
  #else
  // Clean up temp file
  if (rank == 0) {
    unlink("/tmp/nccl_id_simple.dat");
  }
  #endif
  
  return 0;
}
