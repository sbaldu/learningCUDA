#include <iostream>
#include <vector>

__global__ void add(int *a, int *b, int *c) { 
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void print(int* p, int n) {
  for(int i{}; i < n; ++i) {
	std::cout << p[i] << '\n';
  }
}

int main() {
  int const N{512};
  // Initialize data on host
  std::vector<int> a; 
  std::vector<int> b; 
  a.resize(N);
  b.resize(N);
  for(int i{}; i < N; ++i) {
	a[i] = i;
	b[i] = 2*i;
  }
  int* c;
  int size = sizeof(int) * N;
  c = (int*)malloc(size);

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data from host to device
  cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

  // Do the calculation on the device
  add<<<1, N>>>(d_a, d_b, d_c);

  // Copy the results from the device to the host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  print(c, N);

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(c);
}
