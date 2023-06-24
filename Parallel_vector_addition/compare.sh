#

# Compile the serial algorithm
g++ vec_add.cc -o serial.out
# Compile the parallel algorithm
nvcc templated_vec_add.cu -o cuda.out

time ./serial.out
time ./cuda.out
