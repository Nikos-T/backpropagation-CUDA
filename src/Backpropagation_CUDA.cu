#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
#include <cuda.h>
#include <curand_kernel.h>

typedef struct {
	unsigned int seed;
	unsigned int size;
	float *array;
}init_rand_struct;

void * init_rand_t(void *arg) {
	init_rand_struct *a = (init_rand_struct *)arg;
	for (unsigned int i=0; i<a->size; i++) {
		a->array[i] = ((float)rand_r(&(a->seed)))/((float)RAND_MAX);
	}
	pthread_exit(0);
}

// http://aresio.blogspot.gr/2011/05/cuda-random-numbers-inside-kernels.html
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
		// printf("block_id=%u, tid=%u\n", block_id, tid);
		array[block_id*block_size+tid] = curand_uniform(&localState);
		global_states[block_id*block_size+tid] = localState;
	}
}

int main(int argc, char **argv) {
/*	Usage:
*	
*	
*	
*	
*/
unsigned int L = atoi(argv[1]), *layer_sizes = (unsigned int *)malloc(L*sizeof(unsigned int));
FILE *fp;

{	// Parse arguments
if (layer_sizes==NULL) {
	printf("Could not allocate memory to layer_sizes.\nExiting...\n");
	return -1;
}
if ((argc != L+2) && (argc != 2)) {
	printf("Usage:\n./name_of_program L l1 l2 l3 ... lL\nOR\n./name_of_program L\nWhere\nL is the length of the neural network\nand\nl1 l2 l3 ... lL the size of the corresponding layer.\nIf no layer sizes are passed then the program will attempt to read them from \"../data/layer_sizes.mydata\"\nExiting...\n");
	return 1;
}
if (argc == L+2) {
	for (unsigned int i=2; i<argc; i++) {
		layer_sizes[i-2] = atoi(argv[i]);
	}
} else {
	fp = fopen("../data/layer_sizes.mydata", "r");
	if (fp == NULL) {
		printf("Error opening file layer_sizes.mydata.\nExiting...\n");
		return -1;
	}
	fread(layer_sizes, sizeof(unsigned int), L, fp);
	fclose(fp);
}
}

/*
{	// Init rand neural network with pthreads
float **weights = (float **)malloc(L*sizeof(float *)), **biases = (float **)malloc(L*sizeof(float *));
if (weights == NULL) {
	printf("Could not allocate memory to weights.\nExiting...\n");
	return -1;
}
if (biases == NULL) {
	printf("Could not allocate memory to biases.\nExiting...\n");
	return -1;
}

init_rand_struct *t_w_args = (init_rand_struct *)malloc((L-1)*sizeof(init_rand_struct));
init_rand_struct *t_b_args = (init_rand_struct *)malloc((L-1)*sizeof(init_rand_struct));
pthread_t *t_weights = (pthread_t *)malloc((L-1)*sizeof(pthread_t));
pthread_t *t_biases = (pthread_t *)malloc((L-1)*sizeof(pthread_t));
for (unsigned int i=0; i<L-1; i++) {
	weights[i] = (float *)malloc(layer_sizes[i]*layer_sizes[i+1]*sizeof(float));
	biases[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	if (weights[i] == NULL) {
		printf("Could not allocate memory to weights[%u].\nExiting...\n", i);
		return -1;
	}
	if (biases[i] == NULL) {
		printf("Could not allocate memory to biases[%u].\nExiting...\n", i);
		return -1;
	}
	t_w_args[i].seed = time(NULL)+10*i;
	t_w_args[i].size = layer_sizes[i]*layer_sizes[i+1];
	t_w_args[i].array = weights[i];
	pthread_create(&t_weights[i], NULL, init_rand_t, &t_w_args[i]);
	t_b_args[i].seed = time(NULL)+100*i;
	t_b_args[i].size = layer_sizes[i+1];
	t_b_args[i].array = biases[i];
	pthread_create(&t_biases[i], NULL, init_rand_t, &t_b_args[i]);
}



for (unsigned int i=0; i<L-1; i++) {
	pthread_join(t_weights[i], NULL);
	pthread_join(t_biases[i], NULL);
}


// debug !OK working

// for (unsigned int i=0; i<L-1; i++) {
	// printf("Weights%u,%u:\n", i+1, i);
	// for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		// for (unsigned int x=0; x<layer_sizes[i]; x++) {
			// printf("%.2f,", weights[i][y*layer_sizes[i]+x]);
		// }
		// printf("\n");
	// }
	// printf("\nBiases%u:\n", i+1);
	// for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		// printf("%.2f\n", biases[i][y]);
	// }
	// printf("\n");
}
*/


{	// Init rand neural network with CUDA
float **weights = (float **)malloc(L*sizeof(float *)), **biases = (float **)malloc(L*sizeof(float *));
float *weight_D, *bias_D;
if (weights == NULL) {
	printf("Could not allocate memory to weights.\nExiting...\n");
	return -1;
}
if (biases == NULL) {
	printf("Could not allocate memory to biases.\nExiting...\n");
	return -1;
}

// I could take the 2 max values instead of only one.
unsigned int max_layer=0;
for (unsigned int i=0; i<L; i++) {
	if (max_layer<layer_sizes[i]) {
		max_layer = layer_sizes[i];
	}
}
curandState *global_states;
if (cudaMalloc((void **)&global_states, max_layer*max_layer*sizeof(curandState)) != cudaSuccess) {
	printf("Could not allocate gpu memory to global_states.\nExiting...\n");
	return -2;
}
dim3 grid((max_layer+31)/32, (max_layer+31)/32);
dim3 block(32, 32, 1);
setup_rand_Kernel<<<grid, block>>>(global_states, time(NULL));

for (unsigned int i=0; i<L-1; i++) {
	weights[i] = (float *)malloc(layer_sizes[i]*layer_sizes[i+1]*sizeof(float));
	if (cudaMalloc((void **)&weight_D,layer_sizes[i]*layer_sizes[i+1]*sizeof(float)) != cudaSuccess) {
		printf("Could not allocate gpu memory to weight_D.\nExiting...\n");
		return -2;
	}
	biases[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	if (cudaMalloc((void **)&bias_D, layer_sizes[i+1]*sizeof(float)) != cudaSuccess) {
		printf("Could not allocate gpu memory to bias_D.\nExiting...\n");
		return -2;
	}
	if (weights[i] == NULL) {
		printf("Could not allocate memory to weights[%u].\nExiting...\n", i);
		return -1;
	}
	if (biases[i] == NULL) {
		printf("Could not allocate memory to biases[%u].\nExiting...\n", i);
		return -1;
	}
	grid.x = (layer_sizes[i]+31)/32;
	grid.y = (layer_sizes[i+1]+31)/32;
	generate_rand_Kernel<<<grid, block>>>(weight_D, layer_sizes[i]*layer_sizes[i+1], global_states);
	generate_rand_Kernel<<<(layer_sizes[i+1]+31)/32, block>>>(bias_D, layer_sizes[i+1], global_states);
	cudaMemcpy(weights[i], weight_D, layer_sizes[i]*layer_sizes[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(biases[i], bias_D, layer_sizes[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(weight_D);
	cudaFree(bias_D);
	printf("\n========================\n");
}

for (unsigned int i=0; i<L-1; i++) {
	printf("Weights%u,%u:\n", i+1, i);
	for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		for (unsigned int x=0; x<layer_sizes[i]; x++) {
			printf("%.2f,", weights[i][y*layer_sizes[i]+x]);
		}
		printf("\n");
	}
	printf("\nBiases%u:\n", i+1);
	for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		printf("%.2f\n", biases[i][y]);
	}
	printf("\n");
}
}

}