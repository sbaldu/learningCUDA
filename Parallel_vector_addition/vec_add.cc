#include <cassert>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
void add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c, int n) {
  for (int i{}; i < n; ++i) {
	c[i] = a[i] + b[i];
  }
}

template <typename T>
void random_ints(std::vector<T>& vec, int n) {
  for (int i{}; i < n; ++i) {
    vec[i] = std::rand();
  }
}

template <typename T>
void verify(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c, int n) {
  for (int i{}; i < n; ++i) {
    assert(c[i] == a[i] + b[i]);
  }

  std::cout << "Success!\n";
}

int main() {
  // Define the total number of elements
  const int N{1 << 26};

  // Allocate memory on host
  std::vector<int> a(N), b(N), c(N);
  int size{sizeof(int) * N};

  // Initialize input
  random_ints(a, N);
  random_ints(b, N);

  // Sum the two vectors
  add(a, b, c, N);

  // Verify that the result is correct
  verify(a, b, c, N);
}
