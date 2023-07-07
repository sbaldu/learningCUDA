
export CUDA := nvcc
CXX_FLAGS := -O3 -std=c++11

all: ./*.cu
	$(CUDA) $(CXX_FLAGS) mat_sum.cu -o basic.out ;
	$(CUDA) $(CXX_FLAGS) shared_mat_sum.cu -o shared.out ;
	clear ;

basic: mat_sum.cu
	$(CUDA) $(CXX_FLAGS) mat_sum.cu -o basic.out ;
	clear ;

shared: shared_mat_sum.cu
	$(CUDA) $(CXX_FLAGS) shared_mat_sum.cu -o shared.out ;
	clear ;

clean:
	rm -rf ./*.out ;
	clear ;
