#

# Compile the non shared and shared implementation of the algorithm
nvcc ../../ReverseVec/reverse.cu -o basic
nvcc ./sharedReverse.cu -o shared

# Compare the two execution times
time ./basic
time ./shared

# Remove the executables
rm basic
rm shared
