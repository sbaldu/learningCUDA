#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

// Basic kernel for sum of two matrices
__global__ void matSum(const int* a, const int* b, int* c, int n) {
  unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

  unsigned int global_index{col + n * row};
  if (col < n && row < n) {
	c[global_index] = a[global_index] + b[global_index];
  }
}

__host__ void initializeData(std::vector<int>& a, std::vector<int>& b, int n) {
  for (int i{}; i < n * n; ++i) {
    a[i] = rand();
    b[i] = rand();
  }
}

__host__ void verify(std::vector<int> const& a, std::vector<int> const& b, std::vector<int> const& c) {
  for (int i{}; i < a.size(); ++i) {
    assert(a[i] + b[i] == c[i]);
  }

  std::cout << "yay funziona! \n";
}

int main() {
  // Define dimensions of blocks and grids
  const int N{1 << 10};
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_size, grid_size);

  const int size{sizeof(int) * N * N};

  // Allocate memory on host
  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N);

  initializeData(h_a, h_b, N);

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

  // Execute kernel
  matSum<<<grid, block>>>(d_a, d_b, d_c, N);

  // Copy data back to host
  cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

  // Verify result
  verify(h_a, h_b, h_c);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
