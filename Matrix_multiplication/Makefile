
export CUDA := nvcc
CXX_FLAGS := -O3 -std=c++11 -g -G

all: ./*.cu
	$(CUDA) $(CXX_FLAGS) ./mat_mult.cu -o basic.out ;
	$(CUDA) $(CXX_FLAGS) ./shared_mat_mult.cu -o shared.out ;
	$(CUDA) $(CXX_FLAGS) ./rectangular.cu -o rect.out ;
	clear ;

basic: mat_mult.cu
	$(CUDA) $(CXX_FLAGS) ./mat_mult.cu -o basic.out ;
	clear ;

shared: shared_mat_mult.cu
	$(CUDA) $(CXX_FLAGS) ./shared_mat_mult.cu -o shared.out ;
	clear ;

rect: rectangular.cu
	$(CUDA) $(CXX_FLAGS) ./rectangular.cu -o rect.out ;
	clear ;

# scalar: scalar_product.cu
# 	$(CUDA) $(CXX_FLAGS) ./scalar_product.cu -o scalar.out ;

clean:
	rm -rf ./*.out ;
	clear ;
