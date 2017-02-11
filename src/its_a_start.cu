#include <math.h>
#include <cuda.h>

/*
*	L is the size of the neural network
*
*	x is the input				size(x) = size(layer0)=layerSizes[0]
*	y is the desired output		size(y) = size(layerL)=layerSizes[n_layers-1]
*	each * w has the weights of layer l
*	each * b has the biases of layer l
*	dw is the dC/dw
*	db is the dC/db
*	
*	For each step each block is assigned a neuron. So for step l we need layerSizes[step(l)] blocks
*	For each step l we need 
*   	Forward pass: layerSizes[step(l-1)] threads per block to do the row_j*column_j+b_j multiplication
*		Bachward pass: layerSizes[step(l+1)] threads per block to do the column_j*column_j multiplication
*	where j is the neuron (block)
*
*/

template <unsigned int L, unsigned int max_layer_size>
__global__ backPropagationCUDA(float **w, float **b, float *x, float *y, float ** dw, float **db, int *layerSizes) {
	
	__shared__ float partial_sums[1024];
	
	int nid = blockIdx.x/*.y.z*/;
	int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	float z[L];	//it can be smaller.
	int a[L];
	a[0] = x[tid];
	//forward pass
	for (int i=1; i<L; i++) {
		partial_sums[tid] = 0;
		if (tid < layerSizes[i-1] && nid < layerSizes[i]) {
			partial_sums[tid] = w[i][nid*layerSizes[i-1]+tid]*a[i-1];
		} else {
			z[i]=NAN;
			a[i]=NAN;
		}
		__syncthreads();
		if (max_layer_size > 512) { if (tid < 512) { partial_sums[tid] += partial_sums[tid+512]; } __syncthreads; }	//or layerSizes[i-1]
		if (max_layer_size > 256) { if (tid < 256) { partial_sums[tid] += partial_sums[tid+256]; } __syncthreads; }
		if (max_layer_size > 128) { if (tid < 128) { partial_sums[tid] += partial_sums[tid+128]; } __syncthreads; }
		if (max_layer_size >  64) { if (tid <  64) { partial_sums[tid] += partial_sums[tid+ 64]; } __syncthreads; }	//or layerSizes[i-1]
		if (max_layer_size >  32) { if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads; }
		if (max_layer_size >  16) { if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads; }
		if (max_layer_size >   8) { if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads; }
		if (max_layer_size >   4) { if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads; }
		if (tid < 2) { partial_sums[tid]+=partial_sums[tid+2]; } __syncthreads;
		if (tid < 1) {
			z[i] = partial_sums[0] + partial_sums[1];
			a[i]=1/(1+exp(-z[i]));
			dw[tid][i]=z[i];
			db[tid][i]=a[i];
		}
		
	}
	ks
}

int main(int argc, char** argv) {
	
	float **ws, **bs, **dw, **db;
	
	
	cudaMalloc((void **)&ws, (3*4+5*3+4*5+2*4)*sizeof(float));
	cudaMalloc((void **)&bs, (3+5+4+2)*sizeof(float));
	cudaMalloc((void **)&dw, 5*5*sizeof(float));
	cudaMalloc((void **)&db, 5*5*sizeof(float));
	
	ws[0] = {};
	ws[1] = {	0.9, 0.1, 0.7, 0.0,
				0.4, 0.4, 0.9, 0.8,
				0.8, 0.9, 0.6, 0.9};
	
	ws[2] = {	0.6, 0.1, 0.0,
				0.7, 0.7, 0.8,
				0.7, 0.0, 0.6,
				0.3, 0.2, 0.3
				0.6, 0.0, 0.9};
	
	ws[3] = {	0.0, 0.7, 0.6, 0.6, 0.4,
				0.4, 0.1, 0.7, 0.6, 0.9,
				0.3, 0.4, 0.7, 0.1, 0.3,
				0.7, 0.4, 0.2, 0.1, 0.5};
	
	ws[4] = {	0.2, 0.2, 0.6, 0.9,
				0.7, 0.5, 0.8, 0.5};
	
	bs[0] = {};
	bs[1] = {0.2, 0.8, 0.2};
	bs[2] = {0.4, 0.3, 0.8, 0.5, 0.5};
	bs[3] = {0.9, 0.2, 0.7, 0.7};
	bs[4] = {0.3, 0.5};
	
	float x[4] = {0.1, 0.1, 0.2, 0.8};
	
	int layerSizes[5] = {4, 3, 5, 4, 2}
	
	unsigned int L=5, maxl=5;
	float z[5][5], a[5][5];
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(1, 1, maxl);
	backPropagationCUDA<L, maxl><<<blocksPerGrid, threadsPerBlock>>>(ws, bs, x, NULL, dw, db, layerSizes);
	cudaMemcpy(z, dw, 25*sizeof(float));
	cudaMemcpy(a, db, 25*sizeof(float));
	printf("[");
	for (int i=0; i<5; i++) {
		for (int j=0; j<5; j++) {
			printf("%f ", z[j][i]);
		}
		printf("\n");
	}
	printf("]");
	
	
	
	
	
	
}