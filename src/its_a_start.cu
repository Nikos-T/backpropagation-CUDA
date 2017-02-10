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
__global__ backPropagationCUDA(float const ** const w, float const ** const b, float const * const x, float const * const y, float ** dw, float **db, int *layerSizes) {
	
	__shared__ float z[L], partial_sums[1024];
	
	int nid = blockIdx.x/*.y.z*/;
	int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	int a = x[tid];
	//forward pass
	for (int i=1; i<L; i++) {
		partial_sums[tid] = 0;
		if (tid < layerSizes[i-1] && nid < layerSizes[i]) {
			partial_sums[tid] = w[i][nid*layerSizes[i-1]+tid]*a[tid];
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
		}
		
	}

	
	
	
	
	
}