#include <unistd.h>
#include <stdio.h>
#include <pthread.h>

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

{	// Init rand neural network
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


// debug
for (unsigned int i=0; i<L-1; i++) {
	printf("Weights%u,%u:\n", i+1, i);
	for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		for (unsigned int x=0; x<layer_sizes[i]; x++) {
			printf("%.2f,", weights[y*layer_sizes[i]+x]);
		}
		printf("\n");
	}
	printf("\nBiases%u:\n", i+1);
	for (unsigned int y=0; y<layer_sizes[i+1]; y++) {
		printf("%.2f\n", biases[y]);
	}
	printf("\n");
}
}



}

