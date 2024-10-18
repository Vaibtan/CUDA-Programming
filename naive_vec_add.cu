#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EXIT_SIG 0

__global__ void vector_add(int *__restrict vec_a, int *__restrict vec_b, int *__restrict vec_c, int VEC_LEN) {
    // GLOBAL THREAD ID
    int thrd_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thrd_id < VEC_LEN) vec_c[thrd_id] = vec_a[thrd_id] + vec_b[thrd_id];
}

inline void constexpr vector_init(int *vec_a, int VEC_LEN) {
    for (int __iter = 0; __iter < VEC_LEN; ++__iter) vec_a[__iter] = rand() % 100;
}

inline void constexpr error_check(int *vec_a, int *vec_b, int *vec_c, int VEC_LEN) {
    for (int __iter = 0; __iter < VEC_LEN; ++__iter)
        assert(vec_c[__iter] == vec_a[__iter] + vec_b[__iter]);
}

int main() {
    const int static N = 1 << 16;
    int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    size_t bytes = sizeof(int) * N;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector_init(h_a, N);
    vector_init(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //THREAD_BLOCK SIZE, GRID SIZE
    const int static NUM_THREADS = 256, NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

    // LAUNCH KERNEL ON DEFAULT STREAM W/O SHARED_MEM
    vector_add<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    error_check(h_a, h_b, h_c, N);
    printf("COMPLETED SUCCESSFULLY\n");
    return EXIT_SIG;
}