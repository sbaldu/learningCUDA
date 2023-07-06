#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

__global__ void histogram(const int* input, int* output) {
  uint32_t global_index{blockDim.x * blockIdx.x + threadIdx.x};
  uint32_t shared_index{threadIdx.x};

  extern __shared__ int temp[];

  // Set to zero the values in all the bins
  temp[shared_index] = 0;
  // Wait for all the threads to finish up
  __syncthreads();

  // Increase the number of entries in each bin
  atomicAdd(&temp[input[global_index]], 1);
  // Wait for all the threads to finish up
  __syncthreads();

  // Write the content of the histogram from shared memory to global memory
  atomicAdd(&output[threadIdx.x], temp[threadIdx.x]);
}

__host__ void initialize(std::vector<int>& vec) {
  std::mt19937 gen;
  std::uniform_int_distribution<int> dis(0, 255);

  for (auto& x : vec) {
    x = dis(gen);
  }
}

int main() {
  const int N{1 << 10};
  const int block_size{256};
  const int grid_size{N / block_size};
  const int size{N * sizeof(int)};
  const int shared_size{block_size * sizeof(int)};

  std::vector<int> h_in(N);
  std::vector<int> h_out(block_size);
  initialize(h_in);

  // Allocate memory on device
  int *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, shared_size);

  // Copy memory to device
  cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

  // Execute kernel
  histogram<<<grid_size, block_size, shared_size>>>(d_in, d_out);

  // Copy data back to host
  cudaMemcpy(h_out.data(), d_out, shared_size, cudaMemcpyDeviceToHost);

  // Print the content of the histogram
  for (auto x : h_out) {
    std::cout << x << std::endl;
  }

  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
}
