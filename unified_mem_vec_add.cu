#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EXIT_SIG 0

__global__ void vector_add_um(int *__restrict a, int *__restrict b, int *__restrict c, int LEN) {
    // GLOBAL THREAD ID
    int thrd_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thrd_id < LEN) c[thrd_id] = a[thrd_id] + b[thrd_id];
}

inline void constexpr vector_init(int *a, int *b, int LEN) {
    for (int __iter = 0; __iter < LEN; ++__iter) {
        a[__iter] = rand() % 100;
        b[__iter] = rand() % 32;
    }
}

inline void constexpr error_check(int *a, int *b, int *c, int LEN) {
    for (int __iter = 0; __iter < LEN; ++__iter) assert(c[__iter] == a[__iter] + b[__iter]);
}

int main() {
    //device ID for prefetching calls
    int id = cudaGetDevice(&id);
    const int static N = 1 << 16;
    size_t bytes = sizeof(int) * N;
    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    vector_init(a, b, N);

    // data is mostly going to be read from and only occasionally written to
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    const int static BLOCK_SZ = 256;
    const int static GRID_SZ = (int)ceil(N / BLOCK_SZ);

    // LAUNCH KERNEL ON DEFAULT STREAM W/O SHARED_MEM
    vector_add_um<<<GRID_SZ, BLOCK_SZ>>>(a, b, c, N);
    cudaDeviceSynchronize();

    //PREFETCH c TO HOST
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
    error_check(a, b, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("COMPLETED SUCCESSFULLY\n");
    return EXIT_SIG;
}