NVCC=nvcc -arch=sm_20

all:
	$(NVCC) main.cu rands.cu backpropagation.cu -lm -o ../bin/Backpropagation

clean:
	rm -f *~ *.o *.bin ../bin/Backpropagation
