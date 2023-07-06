
export CUDA := nvcc
CXX_FLAGS := -O3 -std=c++11

all: ./*.cu
	$(CUDA) $(CXX_FLAGS) ./histogram.cu -o hist.out ;
	clear ;

clean:
	rm -rf ./*.out ;
	clear ;
