
export CUDA := nvcc
CXX_FLAGS := -O3 -std=c++11

all: ./*.cu
	$(CUDA) $(CXX_FLAGS) vec_add_threads.cu -o add_threads.out ;
	$(CUDA) $(CXX_FLAGS) vec_add_blocks.cu -o add_blocks.out ;
	$(CUDA) $(CXX_FLAGS) vec_add_blocksandthreads.cu -o add_mixed.out ;
	$(CUDA) $(CXX_FLAGS) templated_vec_add.cu -o add_templated.out ;
	clear ;

threads_only: vec_add_threads.cu
	$(CUDA) $(CXX_FLAGS) vec_add_threads.cu -o add_threads.out ;
	clear ;

blocks_only: vec_add_blocks.cu
	$(CUDA) $(CXX_FLAGS) vec_add_blocks.cu -o add_blocks.out ;
	clear ;

mixed_only: vec_add_blocksandthreads.cu
	$(CUDA) $(CXX_FLAGS) vec_add_blocksandthreads.cu -o add_mixed.out ;
	clear ;

templated_only: vec_add_templated.cu
	$(CUDA) $(CXX_FLAGS) templated_vec_add.cu -o add_templated.out ;
	clear ;

clean:
	rm -rf ./*.out ;
	clear ;
