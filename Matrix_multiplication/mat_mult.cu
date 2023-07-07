#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

__global__ void matMul(const int* a, const int* b, int* c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int value{};
  for (int k{}; k < N; ++k) {
	value += a[row * N + k] * b[k * N + col];
  }
  c[col + row * N] = value;
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }

  std::cout << "Success\n";
}

int main() {
  const int N{1 << 5};
  const int size = {N * N * sizeof(int)};

  // Inizialize data on host
  std::vector<int> a(N * N);
  std::vector<int> b(N * N);
  std::vector<int> c(N * N);

  std::generate(a.begin(), a.end(), []() { return rand() % 100; });
  std::generate(b.begin(), b.end(), []() { return rand() % 100; });

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy memory to device
  cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

  // Run kernel
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_size, grid_size);
  matMul<<<grid, block>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  // Copy results back to the host
  cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

  verify_result(a, b, c, N);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
