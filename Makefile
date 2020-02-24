.PHONY: all
all: daxpy mpi_daxpy

daxpy: daxpy.cu cuda_error.h
	nvcc -lcublas -o daxpy daxpy.cu

mpi_daxpy: mpi_daxpy.cc cuda_error.h
	mpic++ -lcudart -lcublas -I$(CUDA_HOME)/include -o mpi_daxpy mpi_daxpy.cc

.PHONY: clean
clean:
	rm -rf daxpy mpi_daxpy

.PHONY: force
force: clean all
