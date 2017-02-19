#include <cuda.h>

/*
template <unsigned int block_size>
__global__ void step_forward(float *weight, float *bias, float *a, float *z, unsigned int prev_layer, unsigned int n_layer, unsigned int max_layer) {
	
	unsigned int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int block_id = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	__shared__ float partial_sums[block_size];
	
	
	
	if (tid < prev_layer) {
		partial_sums[tid] = a[tid]*weight[block_id*max_layer+tid];
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
			z[block_id] = partial_sums[0] + bias[block_id];// + biases[(i-1)*L+block_id];
			a[max_layer+block_id] = 1/(1+expf(-z[block_id]));//1/(1+expf(-z[L*(i-1)+block_id]));
		}
	}
	
}
*/



/*
* 
*
*
*/
template <unsigned int block_size>
__global__ void step_forward1(float *weight, float *bias, float *a, float *z, unsigned int prev_layer_size, unsigned int layer_size) {
	
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int j = (gridDim.x*blockIdx.y + blockIdx.x)*block_size + tid;	//column
	unsigned int i = blockIdx.z;	//row
	__shared__ float partial_sums[block_size];
	
	if (j<prev_layer_size) {
		partial_sums[tid] = a[j]*weight[i*prev_layer_size+j];
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
	if (tid <   1) {
		if (gridDim.x*gridDim.y == 1) {
			partial_sums[0] += partial_sums[1]+bias[i];
			z[i] = partial_sums[0];
			a[i] = 1/(1+expf(-partial_sums[0]));
		} else {
			partial_sums[0] += partial_sums[1];
			z[layer_size*(gridDim.x*blockIdx.y + blockIdx.x)+i] = partial_sums[0];
		}
	}
}

template <unsigned int block_size>
__global__ void step_forward2(float *z, float *bias, float *a, unsigned int layer_size) {
	
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int i = blockIdx.z;
	
	__shared__ float partial_sums[block_size];
	
	partial_sums[tid] = z[tid*layer_size + i];
	
	
	__syncthreads();
	if (block_size >  32) { if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads(); }
	if (block_size >  16) { if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads(); }
	if (block_size >   8) { if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads(); }
	if (block_size >   4) { if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads(); }
	if (block_size >   2) { if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads(); }
	if (tid <   1) {
		partial_sums[0] += partial_sums[1] + bias[i];
		z[i] = partial_sums[0];
		a[i] = 1/(1+expf(-partial_sums[0]));
	}
}

void step_forward_wrapper(float *weight, float *bias, float *a, float *z, unsigned int prev_layer_size, unsigned int layer_size) {
	dim3 grid(1, 1, layer_size);
	dim3 block(1, 1, 1);
	if (prev_layer_size > 1024) {
		//here we need more than step_forward1
		block.x = 32;
		block.y = 32;
		//here z must be of size layer_size*(prev_layer_size/1024)
		if (prev_layer_size/1024 > 32) {	//max_layer<=65535 ~= 1024 * 64
			grid.x = 8;
			grid.y = 8;
			step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
			block.x = 8;
			block.y = 8;
			grid.x = 1;
			grid.y = 1;
			step_forward2<64><<<grid, block>>>(z, bias, a, layer_size);
		} else if (prev_layer_size/1024 > 16) {
			grid.x = 8;
			grid.y = 4;
			step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
			block.x = 8;
			block.y = 4;
			grid.x = 1;
			grid.y = 1;
			step_forward2<32><<<grid, block>>>(z, bias, a, layer_size);
		} else if (prev_layer_size/1024 >  8) {
			grid.x = 4;
			grid.y = 4;
			step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
			block.x = 4;
			block.y = 4;
			grid.x = 1;
			grid.y = 1;
			step_forward2<16><<<grid, block>>>(z, bias, a, layer_size);
		} else if (prev_layer_size/1024 >  4) {
			grid.x = 4;
			grid.y = 2;
			step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
			block.x = 4;
			block.y = 2;
			grid.x = 1;
			grid.y = 1;
			step_forward2<8><<<grid, block>>>(z, bias, a, layer_size);
		} else if (prev_layer_size/1024 >  2) {
			grid.x = 2;
			grid.y = 2;
			step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
			block.x = 2;
			block.y = 2;
			grid.x = 1;
			grid.y = 1;
			step_forward2<4><<<grid, block>>>(z, bias, a, layer_size);
		} else if (prev_layer_size/1024 >  1) {
			grid.x = 2;
			grid.y = 1;
			step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
			block.x = 2;
			block.y = 1;
			grid.x = 1;
			grid.y = 1;
			step_forward2<2><<<grid, block>>>(z, bias, a, layer_size);
		}
	} else if (prev_layer_size > 512) {
		block.x = 32;
		block.y = 32;
		step_forward1<1024><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size > 256) {
		block.x = 32;
		block.y = 16;
		step_forward1<512><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size > 128) {
		block.x = 16;
		block.y = 16;
		step_forward1<256><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >  64) {
		block.x = 16;
		block.y = 8;
		step_forward1<128><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >  32) {
		block.x = 8;
		block.y = 8;
		step_forward1<64><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >  16) {
		block.x = 8;
		block.y = 4;
		step_forward1<32><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >   8) {
		block.x = 4;
		block.y = 4;
		step_forward1<16><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >   4) {
		block.x = 4;
		block.y = 2;
		step_forward1<8><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >   2) {
		block.x = 2;
		block.y = 2;
		step_forward1<4><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	} else if (prev_layer_size >   1) {
		block.x = 2;
		block.y = 1;
		step_forward1<2><<<grid, block>>>(weight, bias, a, z, prev_layer_size, layer_size);
	}
}