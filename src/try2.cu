#include <cuda.h>
#include <stdio.h>




/*	Implementation for max_layer_size = blockDim.x*blockDim.y*blockDim.z
*	block_size a power of 2 */

template <unsigned int block_size>
__global__ void forward_pass(float *weights, float *biases, float *a, float *z, unsigned int *layer_sizes, unsigned int max_layer, unsigned int L) {
	
	unsigned int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int block_id = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	
	__shared__ partial_sums[block_size];
	
	for (int i=1; i<L; i++) {
		if ((block_id >= layer_sizes[i]) && (tid == 0)) {
			a[L*i+block_id] = NAN;
			z[L*(i-1)+block_id] =NAN;
		} else if (tid == 0) {
			a[L*i+block_id] = 0;
			z[L*i+block_id] = 0;
		}
		
		
		
		
		
	}
	
	
	
	
}




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
	FILE *fp;
	
	// Read layer_sizes
	fp = fopen("../data/layerSizes.mydata", "r");
	fread(layer_sizes, sizeof(unsigned int), L, fp);
	fclose(fp);
	cudaMemcpy(layer_sizes_D, layer_sizes, L*sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Read weights
	fp = fopen("../data/weights.mydata", "r");
	fread(weights, sizeof(float), (L-1)*max_layer*max_layer, fp);
	fclose(fp);
	cudaMemcpy(weights_D, weights, (L-1)*max_layer*max_layer*sizeof(float), cudaMemcpyHostToDevice);
	
	//Read biases
	fp = fopen("../data/biases.mydata", "r");
	fread(biases, sizeof(float), (L-1)*max_layer, fp);
	fclose(fp);
	cudaMemcpy(biases_D, biases, (L-1)*max_layer*sizeof(float), cudaMemcpyHostToDevice);
	
	// input-output a (a[0..max_layer_size]=x)
	float *a = (float *)malloc(L*max_layer*sizeof(float));
	float *a_D;
	cudaMalloc((void **)&a_D, L*max_layer*sizeof(float));
	// output z
	float *z = (float *)malloc((L-1)*max_layer*sizeof(float));
	float *z_D;
	cudaMalloc((void **)&z_D, (L-1)*max_layer*sizeof(float));
	
	// Read input
	fp = fopen("../data/input.mydata", "r");
	fread(a, sizeof(float), max_layer, fp);
	fclose(fp);
	cudaMemcpy(a_D, a, max_layer*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 block(4, 2);
	dim3 grid(1, 1, 8);
	
	forward_pass<8><<<grid, block>>>(weights_D, biases_D, a_D, z_D, layer_sizes_D, max_layer, L);
	cudaMemcpy(a, a_D, L*max_layer*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z, z_D, (L-1)*max_layer*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("a=\n");
	for (int i=0; i<max_layer; i++) {
		for (int j=0; j<L; j++) {
			printf("%.2f, ", a[i*L+j]);
		}
		printf("\n");
	}
	printf("z=\n");
	for (int i=0; i<max_layer; i++) {
		for (int j=0; j<L-1; j++) {
			printf("%.2f, ", z[i*L+j]);
		}
		printf("\n");
	}
	
	
}