
export CUDA := nvcc
CXX_FLAGS := -O3 -std=c++11

all: ./*.cu
	$(CUDA) $(CXX_FLAGS) matSum.cu -o basic.out ;
	$(CUDA) $(CXX_FLAGS) sharedMatSum.cu -o shared.out ;
	clear ;

basic: matSum.cu
	$(CUDA) $(CXX_FLAGS) matSum.cu -o basic.out ;
	clear ;

shared: sharedMatSum.cu
	$(CUDA) $(CXX_FLAGS) sharedMatSum.cu -o shared.out ;
	clear ;

clean:
	rm -rf ./*.out ;
	clear ;
