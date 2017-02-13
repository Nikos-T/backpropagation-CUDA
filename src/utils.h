#include <curand_kernel.h>


// rands:
typedef struct {
	unsigned int seed;
	unsigned int size;
	float *array;
}init_rand_struct;
void * init_rand_t(void *arg);

__global__ void setup_rand_Kernel(curandState *global_states, unsigned long seed);
__global__ void generate_rand_Kernel(float *array, unsigned int size, curandState *global_states);

// backpropagation
void step_forward_wrapper(float *weight, float *bias, float *a, float *z, unsigned int prev_layer_size, unsigned int layer_size);