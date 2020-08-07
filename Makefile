.PHONY: all
all: daxpy mpi_daxpy mpienv daxpy_nvtx mpi_daxpy_nvtx

CCFLAGS = -std=c++11
CUDA_HOME ?= $(CUDA_DIR)

daxpy: daxpy.cu cuda_error.h
	nvcc $(CCFLAGS) -lcublas -o daxpy daxpy.cu

daxpy_nvtx: daxpy_nvtx.cu cuda_error.h
	nvcc $(CCFLAGS) -lcublas -lnvToolsExt -o daxpy_nvtx daxpy_nvtx.cu

mpi_daxpy: mpi_daxpy.cc cuda_error.h
	mpic++ $(CCFLAGS) -lcudart -lcublas -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -o mpi_daxpy mpi_daxpy.cc

mpi_daxpy_nvtx: mpi_daxpy_nvtx.cc cuda_error.h
	mpic++ $(CCFLAGS) -lcudart -lcublas -lnvToolsExt -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -o mpi_daxpy_nvtx mpi_daxpy_nvtx.cc

mpienv: mpienv.f90
	mpif90 -o mpienv mpienv.f90

.PHONY: clean
clean:
	rm -rf daxpy mpi_daxpy daxpy_nvtx mpi_daxpy_nvtx

.PHONY: force
force: clean all
