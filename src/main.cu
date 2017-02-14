
#include <stdio.h>
#include <pthread.h>
// #include <cuda.h>
// #include <curand_kernel.h>
#include "utils.h"


int main(int argc, char **argv) {

unsigned int L, *layer_sizes;
FILE *fp;
time_t start, end;
float **weights, **biases, *weight_D, *bias_D;

{	// Parse arguments
if (argc < 2) {
	printf("Usage:\n./name_of_program L l1 l2 l3 ... lL\nOR\n./name_of_program L\nWhere\nL is the length of the neural network\nand\nl1 l2 l3 ... lL the size of the corresponding layer.\nIf no layer sizes are passed then the program will attempt to read them from \"../data/layer_sizes.mydata\"\nExiting...\n");
	return 1;
}
L = atoi(argv[1]);
if ((argc != L+2) && (argc != 2)) {
	printf("Usage:\n./name_of_program L l1 l2 l3 ... lL\nOR\n./name_of_program L\nWhere\nL is the length of the neural network\nand\nl1 l2 l3 ... lL the size of the corresponding layer.\nIf no layer sizes are passed then the program will attempt to read them from \"../data/layer_sizes.mydata\"\nExiting...\n");
	return 1;
}
layer_sizes = (unsigned int *)malloc(L*sizeof(unsigned int));
if (layer_sizes==NULL) {
	printf("Could not allocate memory to layer_sizes.\nExiting...\n");
	return -1;
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
	if (fread(layer_sizes, sizeof(unsigned int), L, fp) != L) {
		printf("Error reading layer_sizes.mydata. Check if number of layers is correct.\nExiting...\n");
		return -1;
	}
	fclose(fp);
}
}



{	// Init rand neural network with pthreads
// start = time(NULL);
weights = (float **)malloc(L*sizeof(float *));
biases = (float **)malloc(L*sizeof(float *));
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
			// printf("%.4f,", weights[i][y*layer_sizes[i]+x]);
		// }
		// printf("\n");
	// }
	// printf("\nBiases%u:\n", i+1);
	// for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		// printf("%.4f\n", biases[i][y]);
	// }
	// printf("\n");
// }
// end = time(NULL);
// printf("Time to compute with pthreads = %u\n", ((unsigned int)(end-start)));
}





/*{	// Init rand neural network with CUDA
start = time(NULL);
weights = (float **)malloc(L*sizeof(float *));
biases = (float **)malloc(L*sizeof(float *));

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
}
// debug OK! working
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
// }
end = time(NULL);
printf("Time to compute with CUDA = %u\n", ((unsigned int)(end-start)));
}*/

// Test forward pass !OK working!
float **a, **z, *x, *a_D, *z_D;
a = (float **)malloc((L-1)*sizeof(float *));
z = (float **)malloc((L-1)*sizeof(float *));
x = (float *)malloc(layer_sizes[0]*sizeof(float));
if (a == NULL) {
	printf("Could not allocate memory to a.\nExiting...\n");
	return -1;
}
if (z == NULL) {
	printf("Could not allocate memory to z.\nExiting...\n");
	return -1;
}
if (x == NULL) {
	printf("Could not allocate memory to x.\nExiting...\n");
	return -1;
}
for (unsigned int i=0; i<L-1; i++) {
	a[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	z[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	if (a[i] == NULL) {
		printf("Could not allocate memory to a[%u].\nExiting...\n", i);
		return -1;
	}
	if (z[i] == NULL) {
		printf("Could not allocate memory to z[%u].\nExiting...\n", i);
		return -1;
	}
}

//Read Input
// fp = fopen("../data/input.mydata", "r");
// if (fp == NULL) {
	// printf("Error opening file input.mydata.\nExiting...\n");
	// return -1;
// }
srand(time(NULL));
for (unsigned int i=0; i<layer_sizes[0]; i++) {
	x[i] = (float)rand()/(float)RAND_MAX;
}
// if (fread(x, sizeof(float), layer_sizes[0], fp) != layer_sizes[0]) {
	// printf("Error reading input.mydata. Check if sizes of layers are correct.\nExiting...\n");
	// return -1;
// }
start = time(NULL);
for (unsigned int i=0; i<L-1; i++) {
	cudaMalloc((void **)&weight_D, layer_sizes[i]*layer_sizes[i+1]*sizeof(float));
	cudaMalloc((void **)&bias_D, layer_sizes[i+1]*sizeof(float));
	if (layer_sizes[i+1]>layer_sizes[i]) {
		cudaMalloc((void **)&a_D, layer_sizes[i+1]*sizeof(float));
		cudaMalloc((void **)&z_D, layer_sizes[i+1]*sizeof(float));		//for layer_size[i]>1024 z needs to be allocated differently
	} else {
		cudaMalloc((void **)&a_D, layer_sizes[i]*sizeof(float));
		cudaMalloc((void **)&z_D, layer_sizes[i]*sizeof(float));
	}
	cudaMemcpy(weight_D, weights[i], layer_sizes[i]*layer_sizes[i+1]*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias_D, biases[i], layer_sizes[i+1]*sizeof(float), cudaMemcpyHostToDevice);
	if (i == 0) {
		cudaMemcpy(a_D, x, layer_sizes[0]*sizeof(float), cudaMemcpyHostToDevice);
	} else {
		cudaMemcpy(a_D, a[i-1], layer_sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
	}
	step_forward_wrapper(weight_D, bias_D, a_D, z_D, layer_sizes[i], layer_sizes[i+1]);
	cudaMemcpy(z[i], z_D, layer_sizes[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(a[i], a_D, layer_sizes[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(weight_D);
	cudaFree(bias_D);
	cudaFree(a_D);
	cudaFree(z_D);
}
end = time(NULL);
printf("time:%u\n", (unsigned int)(end-start));
// test output OK for layer_sizes<1024!
// unsigned int max_layer=0;
// for (unsigned int i=0; i<L; i++) {
	// if (max_layer<layer_sizes[i]) {
		// max_layer = layer_sizes[i];
	// }
// }
// printf("z=\n");
// for (unsigned int i=0; i<max_layer; i++) {
	// for (unsigned int j=0; j<L-1; j++) {
		// if (i < layer_sizes[j+1]) {
			// printf("%.4f,",z[j][i]);
		// } else {
			// printf("       ");
		// }
	// }
	// printf("\n");
// }
// printf("a=\n");
// for (unsigned int i=0; i<max_layer; i++) {
	// for (unsigned int j=0; j<L-1; j++) {
		// if (i < layer_sizes[j+1]) {
			// printf("%.4f,",a[j][i]);
		// } else {
			// printf("       ");
		// }
	// }
	// printf("\n");
// }


// do not forget to free

}