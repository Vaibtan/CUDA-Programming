#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using namespace std;

#define _REP(__x, __n) for (int __x = 0; __x < (__n); ++__x)

__global__ void matmul(const int *a, const int *b, int *c, int N) {
  int __r = blockIdx.y * blockDim.y + threadIdx.y;
  int __c = blockIdx.x * blockDim.x + threadIdx.x;
  c[__r * N + __c] = 0;
  _REP(k, N) c[__r * N + __c] += a[__r * N + k] * b[k * N + __c];
}

inline constexpr auto verify_result = [&](vector<int> &a, vector<int> &b, vector<int> &c, int N) -> void {
  _REP(i, N) _REP(j, N) {
      int tmp = 0;
      _REP(k, N) tmp += a[i * N + k] * b[k * N + j];
      assert(tmp == c[i * N + j]);
  }
};

int main() {
  int N = 1 << 10;
  size_t bytes = N * N * sizeof(int);
  vector<int> h_a(N * N), h_b(N * N), h_c(N * N);

  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int BLOCK_SZ = 32, GRID_SZ = (int) ceil(N / BLOCK_SZ);
  dim3 threads(BLOCK_SZ, BLOCK_SZ);
  dim3 grid(GRID_SZ, GRID_SZ);

  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  verify_result(h_a, h_b, h_c, N);
  cout << "COMPLETED SUCCESSFULLY\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}