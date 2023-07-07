#!/bin/bash

# Compile the non shared and shared implementation of the algorithm
make

# Compare the two execution times
time ./basic.out
time ./shared.out

rm ./basic.out
rm ./shared.out
