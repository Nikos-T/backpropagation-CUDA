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
	
	__shared__ float z[L], partial_sums[max_layer_size];
	
	int nid = blockIdx.x/*.y.z*/;
	int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	//forward pass
	for (int i=1; i<L; i++) {
		if (tid < layerSizes[i-1]) {
			partial_sums[tid] = 
			
			
			
			
			
		}
		
	}
	
	
	
	
	
}