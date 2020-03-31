.PHONY: all
all: daxpy mpi_daxpy mpienv

daxpy: daxpy.cu cuda_error.h
	nvcc -lcublas -o daxpy daxpy.cu

mpi_daxpy: mpi_daxpy.cc cuda_error.h
	mpic++ -lcudart -lcublas -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -o mpi_daxpy mpi_daxpy.cc

mpienv: mpienv.f90
	mpif90 -o mpienv mpienv.f90

.PHONY: clean
clean:
	rm -rf daxpy mpi_daxpy

.PHONY: force
force: clean all
