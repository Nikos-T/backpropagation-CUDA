#include <unistd.h>
#include <stdio.h>


int main(int argc, char **argv) {
/*	Usage:
*	
*	
*	
*	
*/
unsigned int L = atoi(argv[1]), *layer_sizes = (unsigned int *)malloc(L*sizeof(unsigned int));

if (layer_sizes==NULL) {
	printf("Could not allocate memory to layer_sizes.\nExiting...\n");
	return -1;
}

if ((argc != L+2) && (argc != 2)) {
	printf("Usage:\n./name_of_program L l1 l2 l3 ... lL\nOR\n./name_of_program L\nWhere\nL is the length of the neural network\nand\nl1 l2 l3 ... lL the size of the corresponding layer.\nIf no layer sizes are passed then the program will attempt to read them from \"../data/layer_sizes.mydata\"\nExiting...\n");
	return 1;
}

for (int i=2; i<argc; i++) {
	layer_sizes[i-2] = atoi(argv[i]);
}

for (int i=0; i<L; i++) {
	printf("%i\n", layer_sizes[i]);
}



}

