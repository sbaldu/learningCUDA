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

void verify_result(
    const std::vector<int> &a, const std::vector<int> &b, const std::vector<int> &c, int N, int K, int M) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      int tmp = 0;
      for (int k = 0; k < K; k++) {
        // Accumulate the partial results
        tmp += a[i * K + k] * b[k * M + j];
      }

      assert(tmp == c[i * M + j]);
    }
  }

  std::cout << "Success\n";
}

int main() {
  // We are multiplying an N x K matrix with a K x M matrix
  const int N{1 << 10};
  const int K{1 << 9};
  const int M{1 << 10};

  // Inizialize data on host
  std::vector<int> a(N * K);
  std::vector<int> b(K * M);
  std::vector<int> c(N * N);

  // Define the sizes
  const int size_a{N * K * sizeof(int)};
  const int size_b{K * M * sizeof(int)};
  const int size_c{N * M * sizeof(int)};

  std::generate(a.begin(), a.end(), []() { return rand() % 100; });
  std::generate(b.begin(), b.end(), []() { return rand() % 100; });

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  // Create matrices on device
  matrix_t<int> mat_a(N, K, d_a);
  matrix_t<int> mat_b(K, M, d_b);
  matrix_t<int> mat_c(N, M, d_c);

  // Copy memory to device
  cudaMemcpy(mat_a.data, a.data(), size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(mat_b.data, b.data(), size_b, cudaMemcpyHostToDevice);

  // Run kernel
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  dim3 threads(block_size, block_size);
  dim3 blocks(grid_size, grid_size);

  // We must allocate enoughe memory for both the arrays in shared memory
  const int shared_size{2 * block_size * block_size * sizeof(int)};
  matMul<<<blocks, threads, shared_size>>>(mat_a, mat_b, mat_c, block_size);

  // Copy result back to the host
  cudaMemcpy(c.data(), mat_c.data, size_c, cudaMemcpyDeviceToHost);

  verify_result(a, b, c, N, K, M);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
