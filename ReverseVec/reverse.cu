#include <iostream>
#include <numeric>
#include <vector>

#define n 10

__global__ void reverse(int* in, int* out) {
  int index = threadIdx.x;
  out[index] = in[n - index - 1];
}

int main() {
  // Allocate memory on host
  std::vector<int> a(n);
  std::vector<int> b(n);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);

  int size{sizeof(int) * n};

  // Allocate memory on device
  int *d_a, *d_b;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);

  // Move data from host to device
  cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

  // Calculate output on device
  reverse<<<1, n>>>(d_a, d_b);

  // Move output from device to host
  cudaMemcpy(b.data(), d_b, size, cudaMemcpyDeviceToHost);

  // Print output
  for (auto const& x : b) {
    std::cout << x << '\n';
  }

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
}
