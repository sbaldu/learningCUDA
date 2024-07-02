
#include <chrono>
#include <iostream>
#include <random>
#include <span>
#include <vector>

template <typename T>
__global__ void kernel(const T *a, T *b, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    b[i] = a[i];
  }
}

template <typename T, std::size_t N = std::dynamic_extent>
__host__ void fill(std::span<T> container) {
  for (auto &elem : container) {
    elem = static_cast<T>(rand());
  }
}

template <typename T, std::size_t N = std::dynamic_extent>
__host__ void validate(std::span<const T> a, std::span<const T> b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      printf("Validation failed at index %zu\n", i);
      return;
    }
  }
  printf("Validation passed\n");
}

template <typename T>
void pageableRun(std::size_t N) {
  std::vector<T> a(N);
  std::vector<T> b(N);
  fill<T>(a);

  T *d_a, *d_b;
  cudaMalloc(&d_a, N * sizeof(T));
  cudaMalloc(&d_b, N * sizeof(T));

  // this is going to copy memory from pageable to pinned memory first,
  // and then to device memory
  cudaMemcpy(d_a, a.data(), N * sizeof(T), cudaMemcpyHostToDevice);
  const auto blockSize{256};
  const auto gridSize{std::ceil((float)N / blockSize)};
  kernel<<<gridSize, blockSize>>>(d_a, d_b, N);
  // again, from device memory to pinned memory, and then to pageable memory
  cudaMemcpy(b.data(), d_b, N * sizeof(T), cudaMemcpyDeviceToHost);

  //validate<T>(a, b);
}

template <typename T>
void pinnedRun(std::size_t N) {
  T *a, *b;
  cudaMallocHost(&a, N * sizeof(T));
  cudaMallocHost(&b, N * sizeof(T));
  fill<T>(std::span<T>{a, N});

  T *d_a, *d_b;
  cudaMalloc(&d_a, N * sizeof(T));
  cudaMalloc(&d_b, N * sizeof(T));

  // this is going to copy memory from pageable to pinned memory first,
  // and then to device memory
  cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice);
  const auto blockSize{256};
  const auto gridSize{std::ceil((float)N / blockSize)};
  kernel<<<gridSize, blockSize>>>(d_a, d_b, N);
  // again, from device memory to pinned memory, and then to pageable memory
  cudaMemcpy(b, d_b, N * sizeof(T), cudaMemcpyDeviceToHost);

  //validate<T>(a, b);
}

int main() {
  const std::size_t N{1 << 20};
  const std::size_t nruns{10};

  auto start{std::chrono::high_resolution_clock::now()}; 
  auto finish{std::chrono::high_resolution_clock::now()}; 

  std::vector<long long> pag(nruns, 0);
  std::vector<long long> pin(nruns, 0);
  for (size_t i{}; i < nruns; ++i) {
	start = std::chrono::high_resolution_clock::now();
	pageableRun<float>(N);
	finish = std::chrono::high_resolution_clock::now();
	pag.push_back((finish - start).count());
  }
  for (size_t i{}; i < nruns; ++i) {
	start = std::chrono::high_resolution_clock::now();
	pinnedRun<float>(N);
	finish = std::chrono::high_resolution_clock::now();
	pin.push_back((finish - start).count());
  }

  std::cout << "Pageable avg: " << std::accumulate(pag.begin(), pag.end(), 0) / (float)nruns << '\n';
  std::cout << "Pinned avg: " << std::accumulate(pin.begin(), pin.end(), 0) / (float)nruns << '\n';
}
