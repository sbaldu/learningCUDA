#include <cassert>
#include <iostream>
#include <random>
#include <vector>

__global__ void reverse(int* in, int* out, int n) {
  int index = threadIdx.x;
  out[index] = in[n - index - 1];
}

__host__ void initialize(std::vector<int>& vec) {
  for (int i{}; i < vec.size(); ++i) {
    vec[i] = std::rand();
  }
}

__host__ void verify(std::vector<int> const& input, std::vector<int> const& output) {
  size_t n{input.size()};
  for (int i{}; i < n; ++i) {
    assert(input[i] == output[n - i - 1]);
  }

  std::cout << "The result is correct!\n";
}

int main() {
  int const threadsPerBlock{1024};
  int const N{threadsPerBlock};
  int const size{N * sizeof(int)};

  // Allocate memory on host
  std::vector<int> a(N);
  std::vector<int> b(N);
  initialize(a);

  // Allocate memory on device
  int *d_a, *d_b;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);

  // Move data from host to device
  cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

  // Calculate output on device
  reverse<<<1, N>>>(d_a, d_b, N);

  // Move output from device to host
  cudaMemcpy(b.data(), d_b, size, cudaMemcpyDeviceToHost);

  // Verify result
  verify(a, b);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
}
