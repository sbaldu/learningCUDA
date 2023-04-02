#include <iostream>

__global__ void add(int *a, int *b, int *c) { 
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; 
}

#define N 32

void random_ints(int *a) {
  for (int i{}; i < N; ++i) {
    a[i] = rand();
  }
}

int main() {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof(int);

  // Alloc memory on host and initialize data
  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);
  random_ints(a);
  random_ints(b);

  // Alloc memory on device
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Calculate the output on the device
  add<<<N, 1>>>(a, b, c);

  // Copy the output to the host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  int sum{};
  for (int i{}; i < N; ++i) {
    sum += c[i];
  }

  // Free memory on host
  free(a);
  free(b);
  free(c);
  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  std::cout << "The sum is : " << sum << '\n';
}
