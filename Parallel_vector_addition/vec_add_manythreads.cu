#include <iostream>

__global__ void add(int* a, int* b, int* c, int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
	c[index] = a[index] + b[index];
  }
}

#define N 512
#define M 128

void random_ints(int* a) {
  for (int i{}; i < N; ++i) {
	a[i] = rand();
  }
}

int main() {
  // Allocate memory on host
  int *a, *b, *c;
  int size{sizeof(int) * N};
  a = (int*)malloc(size);
  b = (int*)malloc(size);
  c = (int*)malloc(size);
  // Initialize input
  random_ints(a);
  random_ints(b);

  // Allocate memory on device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data from host to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

  // Calculate the output on device
  add<<<N/M, M>>>(d_a, d_b, d_c, N);

  // Copy the output back to host
  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Print the output
  int sum{};
  for(int i{}; i < size; ++i) {
	sum += c[i];
  }
  std::cout << "The sum is : " << sum << '\n';

  // Free memory
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
