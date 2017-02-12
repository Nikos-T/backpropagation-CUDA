#include <cuda.h>
#include <stdio.h>








int main(int argc, char **argv) {
	
	unsigned int L=atoi(argv[1]), max_layer=atoi(argv[2]);
	unsigned int *layer_sizes = (unsigned int *)malloc(L*sizeof(unsigned int));
	float *weights = (float *)malloc((L-1)*max_layer*max_layer*sizeof(float));
	float *biases = (float *)malloc((L-1)*max_layer*sizeof(float));
	unsigned int *layer_sizes_D;
	float *weights_D, biases_D;
	cudaMalloc((void **)&layer_sizes, L*sizeof(unsigned int));
	cudaMalloc((void **)&weights_D, (L-1)*max_layer*max_layer*sizeof(float));
	cudaMalloc((void **)&biases_D, (L-1)*max_layer*sizeof(float));
	
	
	
	
	
}