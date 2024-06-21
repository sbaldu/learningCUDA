cmake_minimum_required(VERSION 3.16.0)

project(learningCUDA LANGUAGES CXX CUDA VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

add_subdirectory(Parallel_vector_addition)
# add_subdirectory(Histogram)
# add_subdirectory(Matrix_multiplication)
# add_subdirectory(ReverseVec)
# add_subdirectory(SharedMemory)
