#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

template <typename T>
struct matrix_t {
  int rows;
  int cols;
  T *data;

  matrix_t(int n_rows, int n_cols, T *data) : rows{n_rows}, cols{n_cols}, data{data} {}

  __host__ __device__ T &operator[](int index) { return data[index]; }
  __host__ __device__ const T &operator[](int index) const { return data[index]; }
};

template <typename T>
__global__ void matMul(const matrix_t<T> a, const matrix_t<T> b, matrix_t<T> c, int block_size) {
  // Allocate arrays on shared memory
  extern __shared__ int shared[];
  int *s_a{shared};
  int *s_b{&shared[block_size * block_size]};

  unsigned int row{blockIdx.y * blockDim.y + threadIdx.y};
  unsigned int col{blockIdx.x * blockDim.x + threadIdx.x};

  // Temporary variable containing the result of the moltiplication
  int temp{};

  for (int i{}; i < a.cols; i += blockDim.x) {
    // Fill the arrays in shared memory
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * a.cols + threadIdx.x + i];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[(threadIdx.y + i) * b.cols + col];

    __syncthreads();

    for (int j{}; j < blockDim.x; ++j) {
      temp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    c[row * b.cols + col] = temp;
  }
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
  const int N{1 << 10};
  const int size{N * N * sizeof(int)};

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

  // Create matrices on device
  matrix_t<int> mat_a(N, N, d_a);
  matrix_t<int> mat_b(N, N, d_b);
  matrix_t<int> mat_c(N, N, d_c);

  // Copy memory to device
  cudaMemcpy(mat_a.data, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(mat_b.data, b.data(), size, cudaMemcpyHostToDevice);

  // Run kernel
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  dim3 threads(block_size, block_size);
  dim3 blocks(grid_size, grid_size);

  // We must allocate enoughe memory for both the arrays in shared memory
  const int shared_size{2 * block_size * block_size * sizeof(int)};
  matMul<<<blocks, threads, shared_size>>>(mat_a, mat_b, mat_c, block_size);

  // Copy result back to the host
  cudaMemcpy(c.data(), mat_c.data, size, cudaMemcpyDeviceToHost);

  verify_result(a, b, c, N);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
