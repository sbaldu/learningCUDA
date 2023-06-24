#include <cassert>
#include <iostream>
#include <random>
#include <vector>

__global__ void reverse(const int* in, int* out, int n) {
  unsigned int th{threadIdx.x};

  // Declare array on shared memory
  extern __shared__ int temp[];

  // Load data in the shared memory array
  temp[th] = in[th];

  // Make sure that all the threads are done building the shared array
  __syncthreads();

  if (th < n) {
    out[th] = temp[n - th - 1];
  }
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
  const int threadsPerBlock{256};
  const int N{threadsPerBlock};
  const int size{N * sizeof(int)};

  std::vector<int> in(N), out(N);
  initialize(in);

  // Allocate memory on device
  int *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copy memory to device, execute kernel and copy result back to host
  cudaMemcpy(d_in, in.data(), size, cudaMemcpyHostToDevice);
  reverse<<<1, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out, N);
  cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost);

  // Verify result
  verify(in, out);

  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
}
