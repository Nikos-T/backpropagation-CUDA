/* rands */
#include <pthread.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "utils.h"

// Pthreads:
void * init_rand_t(void *arg) {
	init_rand_struct *a = (init_rand_struct *)arg;
	for (unsigned int i=0; i<a->size; i++) {
		a->array[i] = ((float)rand_r(&(a->seed)))/((float)RAND_MAX);
	}
	pthread_exit(0);
}

// CUDA http://aresio.blogspot.gr/2011/05/cuda-random-numbers-inside-kernels.html
__global__ void setup_rand_Kernel(curandState *global_states, unsigned long seed) {
	unsigned int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int block_id = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	
	unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
	
	curand_init(seed, block_id*block_size+tid, 0, &global_states[block_id*block_size+tid]);
	
	
}
__global__ void generate_rand_Kernel(float *array, unsigned int size, curandState *global_states) {
	
	unsigned int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int block_id = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	
	unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
	
	if (block_size*block_id+tid < size) {
		curandState localState = global_states[block_id*block_size+tid];
		array[block_id*block_size+tid] = curand_uniform(&localState);
		global_states[block_id*block_size+tid] = localState;
	}
}