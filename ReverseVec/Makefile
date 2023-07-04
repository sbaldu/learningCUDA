
export CUDA := nvcc
CXX_FLAGS := -O3 -std=c++11

all: ./*.cu
	$(CUDA) $(CXX_FLAGS) reverse.cu -o basic.out ;
	$(CUDA) $(CXX_FLAGS) sharedReverse.cu -o shared.out ;
	clear ;

basic: reverse.cu
	$(CUDA) $(CXX_FLAGS) reverse.cu -o basic.out ;
	clear ;

shared: sharedReverse.cu
	$(CUDA) $(CXX_FLAGS) sharedReverse.cu -o shared.out ;
	clear ;

clean:
	rm -rf ./*.out ;
	clear ;
