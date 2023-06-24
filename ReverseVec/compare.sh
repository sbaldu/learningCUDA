#

# Compile the non shared and shared implementation of the algorithm
nvcc reverse.cu -o basic.out
nvcc sharedReverse.cu -o shared.out

# Compare the two execution times
time ./basic.out
time ./shared.out
