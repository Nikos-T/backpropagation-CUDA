#include <cuda.h>
#include <stdio.h>
#include <math.h>



/*	Implementation for max_layer_size = blockDim.x*blockDim.y*blockDim.z
*	block_size a power of 2 */

template <unsigned int block_size>
__global__ void step_forward(float *weights, float *biases, float *a, float *z, unsigned int prev_layer, unsigned int n_layer, unsigned int max_layer) {
	
	unsigned int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int block_id = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	__shared__ float partial_sums[block_size];
	
	if (tid < prev_layer) {
		partial_sums[tid] = a[tid]*weights[block_id*max_layer+tid];
	} else {
		partial_sums[tid] = 0;
	}
	__syncthreads();
	if (block_size > 512) { if (tid < 512) { partial_sums[tid] += partial_sums[tid+512]; } __syncthreads(); }
	if (block_size > 256) { if (tid < 256) { partial_sums[tid] += partial_sums[tid+256]; } __syncthreads(); }
	if (block_size > 128) { if (tid < 128) { partial_sums[tid] += partial_sums[tid+128]; } __syncthreads(); }
	if (block_size >  64) { if (tid <  64) { partial_sums[tid] += partial_sums[tid+ 64]; } __syncthreads(); }
	if (block_size >  32) { if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads(); }
	if (block_size >  16) { if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads(); }
	if (block_size >   8) { if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads(); }
	if (block_size >   4) { if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads(); }
	if (block_size >   2) { if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads(); }
	if (block_size >   1) {
		if (tid <   1) {
			partial_sums[tid] += partial_sums[tid+  1];
			//z[L*(i-1)+block_id] = i;
			//a[L*i+block_id] = i;
			z[block_id] = partial_sums[0] + biases[block_id];// + biases[(i-1)*L+block_id];
			a[max_layer+block_id] = 1/(1+expf(-z[block_id]));//1/(1+expf(-z[L*(i-1)+block_id]));
		}
	}
	
}

int main(int argc, char **argv) {
	
	unsigned int L=atoi(argv[1]), max_layer=atoi(argv[2]);
	unsigned int *layer_sizes = (unsigned int *)malloc(L*sizeof(unsigned int));
	float *weights = (float *)malloc((L-1)*max_layer*max_layer*sizeof(float));
	float *biases = (float *)malloc((L-1)*max_layer*sizeof(float));
	unsigned int *layer_sizes_D;
	float *weights_D, *biases_D;
	cudaMalloc((void **)&layer_sizes_D, L*sizeof(unsigned int));
	cudaMalloc((void **)&weights_D, (L-1)*max_layer*max_layer*sizeof(float));
	cudaMalloc((void **)&biases_D, (L-1)*max_layer*sizeof(float));
	FILE *fp;
	
	//printf("Most mallocs\n");
	
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
	
	printf("allocated a,z \n");
	// Read input
	fp = fopen("../data/input.mydata", "r");
	fread(a, sizeof(float), max_layer, fp);
	fclose(fp);
	cudaMemcpy(a_D, a, max_layer*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 block(4, 2);
	dim3 grid(1, 1, 8);
	
	
	// forward_pass<8><<<grid, block>>>(weights_D, biases_D, a_D, z_D, layer_sizes_D, max_layer, L);
	grid.z=3;
	step_forward<8><<<grid, block>>>(weights_D, biases_D, a_D, z_D, 4, 3, 5);
	grid.z=5;
	step_forward<8><<<grid, block>>>(&weights_D[25], &biases_D[5], &a_D[5], &z_D[5], 3, 5, 5);
	grid.z=4;
	step_forward<8><<<grid, block>>>(&weights_D[50], &biases_D[10], &a_D[10], &z_D[10], 5, 4, 5);
	grid.z=2;
	step_forward<8><<<grid, block>>>(&weights_D[75], &biases_D[15], &a_D[15], &z_D[15], 4, 2, 5);

	cudaMemcpy(a, a_D, L*max_layer*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z, z_D, (L-1)*max_layer*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("a=\n");
	for (int i=0; i<max_layer; i++) {
		for (int j=0; j<L; j++) {
			printf("%.2f, ", a[i+j*max_layer]);
		}
		printf("\n");
	}
	printf("z=\n");
	for (int i=0; i<max_layer; i++) {
		for (int j=0; j<L-1; j++) {
			printf("%.2f, ", z[i+j*max_layer]);
		}
		printf("\n");
	}
	
	
}