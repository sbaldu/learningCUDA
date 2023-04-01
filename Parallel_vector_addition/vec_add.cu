#include <iostream>

__global__ void add(int *a, int *b, int *c) {
  *c = *a + *b;
}

int main() {
	// Initialize data on host
    int a = 2, b = 7; 
	int c; 
	int size = sizeof(int);

	// Allocate memory on device
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	// Copy data from host to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// Do the calculation on the device
    add<<<1,1>>>(d_a, d_b, d_c);

	// Copy the results from the device to the host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "The result is : " << c << '\n';
}
