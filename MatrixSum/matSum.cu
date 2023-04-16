#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#define N 1024 

// Basic kernel for sum of two matrices
__global__ void matSum(const int* a, const int* b, int* c) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  c[i + N * j] = a[i + N * j] + b[i + N * j];
}

__host__ void initializeData(std::vector<int>& a, 
                             std::vector<int>& b) {
  for(int i{}; i < N * N; ++i) {
	a[i] = rand();
	b[i] = rand();
  }
}

__host__ void verify(std::vector<int> const& a,
					 std::vector<int> const& b,
					 std::vector<int> const& c) {
  for(int i{}; i < a.size(); ++i) {
	assert(a[i] + b[i] == c[i]);
  }

  std::cout << "yay funziona! \n";
}

int main() {
  // Define dimensions of blocks and grids
  int const threadsPerBlock{32};
  dim3 block(threadsPerBlock, threadsPerBlock);
  dim3 grid(N / block.x, N / block.y);

  int const size{ sizeof(int) * N * N };

  // Allocate memory on host
  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N);
  
  initializeData(h_a, h_b);

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

  // Execute kernel
  matSum<<<grid, block>>>(d_a, d_b, d_c);

  // Copy data back to host
  cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

  // Verify result
  verify(h_a, h_b, h_c);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
