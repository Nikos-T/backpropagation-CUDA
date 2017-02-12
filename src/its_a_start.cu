#include <math.h>
#include <cuda.h>
#include <stdio.h>
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
// Doesn't work
template <unsigned int threadsPerBlock>
__global__ void backPropagationCUDA(float **w, float **b, const float *x, float *y, float * dw, float *db, unsigned int *layerSizes, unsigned int L) {
	
	__shared__ float partial_sums[threadsPerBlock];
	
	int nid = blockIdx.z;//.y.x
	int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	float z, a;
	a = x[tid];
	//forward pass
	for (int i=1; i<L; i++) {
		partial_sums[tid] = 0;
		if (tid < layerSizes[i-1] && nid < layerSizes[i]) {
			//printf("thread %i from block %i\n", tid, nid);
			partial_sums[tid] = w[i][nid*layerSizes[i-1]+tid]*a;
		} else {
			z=NAN;
			a=NAN;
		}
		__syncthreads();
		if (threadsPerBlock > 512) { if (tid < 512) { partial_sums[tid] += partial_sums[tid+512]; } __syncthreads(); }	//or layerSizes[i-1]
		if (threadsPerBlock > 256) { if (tid < 256) { partial_sums[tid] += partial_sums[tid+256]; } __syncthreads(); }
		if (threadsPerBlock > 128) { if (tid < 128) { partial_sums[tid] += partial_sums[tid+128]; } __syncthreads(); }
		if (threadsPerBlock >  64) { if (tid <  64) { partial_sums[tid] += partial_sums[tid+ 64]; } __syncthreads(); }	//or layerSizes[i-1]
		if (threadsPerBlock >  32) { if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads(); }
		if (threadsPerBlock >  16) { if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads(); }
		if (threadsPerBlock >   8) { if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads(); }
		if (threadsPerBlock >   4) { if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads(); }
		if (tid < 2) { partial_sums[tid]+=partial_sums[tid+2]; } __syncthreads();
		if (tid < 1) {
			z = partial_sums[0] + partial_sums[1];
			a=1/(1+exp(-z));
			dw[nid*L + i]=z;
			db[nid*L + i]=a;
			printf("\n\nfadsgdsafgafd=%i\n\n", i);
		}
		
	}
	
}


int main(int argc, char** argv) {
	
	/* Read L */
	unsigned int L = atoi(argv[1]), maxLayer=0, *layerSizes, *layerSizesD;
	float **weights, **weightsD, **biases, **biasesD, **sup_wei, **sup_bia, *x, *xD, *dw, *dwD, *db, *dbD;
	FILE *fp;
	
	/* Some mem alloc */
	layerSizes = (unsigned int *)malloc(L*sizeof(unsigned int));
	cudaMalloc((void **)&layerSizesD, L*sizeof(unsigned int));
	weights = (float **)malloc(L*sizeof(float *));
	cudaMalloc((void **)&weightsD, L*sizeof(float *));
	biases = (float **)malloc(L*sizeof(float *));
	cudaMalloc((void **)&biasesD, L*sizeof(float *));
	sup_wei = (float **)malloc(L*sizeof(float *));
	sup_bia = (float **)malloc(L*sizeof(float *));
	//https://devtalk.nvidia.com/default/topic/410182/double-pointer-allocation/?offset=5
	cudaMemcpy(sup_wei, weightsD, L*sizeof(float *), cudaMemcpyDeviceToHost);
	cudaMemcpy(sup_bia, biasesD, L*sizeof(float *), cudaMemcpyDeviceToHost);
	
	// Read layerSizes
	fp = fopen("../data/layerSizes.mydata", "r");
	fread(layerSizes, sizeof(unsigned int), L, fp);
	fclose(fp);
	cudaMemcpy(layerSizesD, layerSizes, L*sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	for (int i=0; i<L; i++) if (maxLayer<layerSizes[i]) maxLayer = layerSizes[i];
	
	// More mem alloc
	dw = (float *)malloc(L*maxLayer*sizeof(float));
	db = (float *)malloc(L*maxLayer*sizeof(float));
	cudaMalloc((void **)&dwD, L*maxLayer*sizeof(float));
	cudaMalloc((void **)&dwD, L*maxLayer*sizeof(float));
	x = (float *)malloc(layerSizes[0]*sizeof(float));
	cudaMalloc((void **)&xD, layerSizes[0]*sizeof(float));
	for (int i=1; i<L; i++) {
		weights[i] = (float *)malloc(layerSizes[i]*layerSizes[i-1]*sizeof(float));
		cudaMalloc((void **)&sup_wei[i], layerSizes[i]*layerSizes[i-1]*sizeof(float));
		biases[i] = (float *)malloc(layerSizes[i]*sizeof(float));
		cudaMalloc((void **)&sup_bia[i], layerSizes[i]*sizeof(float));
	}
	
	//printf("Memory allocation complete!\n");
	// Read weights
	fp = fopen("../data/weights.mydata", "r");
	
	for (int i=1; i<L; i++) {
		fread(weights[i], sizeof(float), layerSizes[i-1]*layerSizes[i], fp);
		cudaMemcpy(sup_wei[i], weights[i], layerSizes[i-1]*layerSizes[i]*sizeof(float), cudaMemcpyHostToDevice);
	}
	fclose(fp);
	
	
	
	// Read biases
	fp = fopen("../data/biases.mydata", "r");
	for (int i=1; i<L; i++) {
		fread(biases[i], sizeof(float), layerSizes[i], fp);
		cudaMemcpy(sup_bia[i], biases[i], layerSizes[i]*sizeof(float), cudaMemcpyHostToDevice);
	}
	fclose(fp);
	
	
	
	// Check if data was read debug !OK!
	/*
	cudaMemcpy(layerSizes, layerSizesD, L*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for (int i=1; i<L; i++) {
		cudaMemcpy(biases[i], sup_bia[i], layerSizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(weights[i], sup_wei[i], layerSizes[i]*layerSizes[i-1]*sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	
	printf("LayerSizes: ");
	for (int i=0; i<L; i++) {
		printf("%u, ", layerSizes[i]);
	}
	printf("\n\nBiases:\n");
	for (int i=1; i<L; i++) {
		for (int j=0; j<layerSizes[i]; j++) {
			printf("%.1f, ", biases[i][j]);
		}
		printf("\n");
	}
	printf("\nWeights:\n");
	for (int i=1; i<L; i++) {
		for (int j=0; j<layerSizes[i]; j++) {
			for (int k=0; k<layerSizes[i-1]; k++) {
				printf("%.1f, ",weights[i][j*layerSizes[i-1]+k]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	*/
	dim3 block(8,1, 1);
	dim3 grid(1,1,5);
	
	//input:
	x[0] = 0.1;
	x[1] = 0.1;
	x[2] = 0.2;
	x[3] = 0.8;
	cudaMemcpy(xD, x, layerSizes[0]*sizeof(float), cudaMemcpyHostToDevice);
	//actually 5 is the max layer
	backPropagationCUDA<8><<<grid, block>>>(weightsD, biasesD, xD, NULL, dwD, dbD, layerSizesD, L);
	cudaMemcpy(dw, dwD, L*maxLayer*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(db, dbD, L*maxLayer*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("\nZ:\n");
	for (int i=0; i<maxLayer; i++) {
		for (int j=0; j<L; j++) {
			printf("%.2f, ", dw[i*L+j]);
		}
		printf("\n");
	}
	printf("\na:\n");
	for (int i=0; i<maxLayer; i++) {
		for (int j=0; j<L; j++) {
			printf("%.2f, ", db[i*L+j]);
		}
		printf("\n");
	}
	
	free(dw);
	free(db);
	free(layerSizes);
	free(weights);
	free(biases);
	free(sup_wei);
	free(sup_bia);
	cudaFree(dwD);
	cudaFree(dbD);
	cudaFree(layerSizesD);
	cudaFree(weightsD);
	cudaFree(biasesD);
	
	
	
}