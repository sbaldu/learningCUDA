#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

__global__ void cube(int* a, int* b) { b[threadIdx.x] = std::pow(a[threadIdx.x], 3); }

int main() {
  int const threadsPerBlock{256};
  int const N{32 * 256};
  int const size{sizeof(int) * N};

  // Allocate memory on host
  std::vector<int> in(N);
  std::vector<int> out(N);
  std::iota(in.begin(), in.end(), 1);

  // Allocate memory on device
  int* d_in;
  int* d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copy memory to device
  cudaMemcpy(d_in, in.data(), size, cudaMemcpyHostToDevice);

  // Calculate the cubes and save them on the output pointer
  cube<<<N / threadsPerBlock, threadsPerBlock>>>(d_in, d_out);

  // Copy the output back to host
  cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost);

  // Calculate the sum
  long long int sum{};
  for (auto const& x : out) {
    sum += x;
  }
  std::cout << "The final output is : " << sum << '\n';

  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
}
