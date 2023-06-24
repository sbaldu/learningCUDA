#include <cassert>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
__global__ void add(const T* a, const T* b, T* c, int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

template <typename T>
void random_ints(std::vector<T>& vec, int n) {
  for (int i{}; i < n; ++i) {
    vec[i] = std::rand();
  }
}

template <typename T>
void verify(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c, int n) {
  for (int i{}; i < n; ++i) {
    assert(c[i] == a[i] + b[i]);
  }

  std::cout << "Success!\n";
}

int main() {
  // Define the total number of elements
  const int N{512};
  const int M{128};

  // Allocate memory on host
  std::vector<int> a(N), b(N), c(N);
  int size{sizeof(int) * N};
  // Initialize input
  random_ints(a, N);
  random_ints(b, N);

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data from host to device
  cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

  // Calculate the output on device
  add<<<N / M, M>>>(d_a, d_b, d_c, N);

  // Copy the output back to host
  cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

  // Verify that the result is correct
  verify(a, b, c, N);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
