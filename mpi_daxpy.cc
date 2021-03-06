/*
 * =====================================================================================
 *
 *       Filename:  mpi_daxpy.c
 *
 *    Description:  Adds MPI to cublas test, to debug issue on Summit
 *
 *        Version:  1.0
 *        Created:  05/20/2019 10:33:30 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "cuda_runtime_api.h"

#define GPU_CHECK_CALLS
#include "cuda_error.h"

// column major
#define IDX2C(i,j,ld) (((j)*(ld))+(i))


static cublasHandle_t handle;


void set_rank_device(int n_ranks, int rank) {
    int n_devices, device, ranks_per_device;
    size_t memory_per_rank;
    cudaDeviceProp device_prop;

    CHECK("get device count", cudaGetDeviceCount(&n_devices));

    if (n_ranks > n_devices) {
        if (n_ranks % n_devices != 0) {
            printf("ERROR: Number of ranks (%d) not a multiple of number of GPUs (%d)\n",
                   n_ranks, n_devices);
            exit(EXIT_FAILURE);
        }
        ranks_per_device = n_ranks / n_devices;
        device = rank / ranks_per_device;
    } else {
        ranks_per_device = 1;
        device = rank;
    }

    CHECK("get device props", cudaGetDeviceProperties(&device_prop, device));
    memory_per_rank = device_prop.totalGlobalMem / ranks_per_device;
    printf("RANK[%d/%d] => DEVICE[%d/%d] mem=%zd\n", rank+1, n_ranks,
           device+1, n_devices, memory_per_rank);

    CHECK("set device", cudaSetDevice(device));
}


int main(int argc, char **argv) {
    int n = 1024;
    int world_size, world_rank;

    double a = 2.0;
    double sum = 0.0;

    double *x, *y, *d_x, *d_y;
    double *m_x, *m_y;

    char *mb_per_core;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    x = (double *)malloc(n*sizeof(*x));
    if (x == NULL) {
        printf("host malloc(x) failed\n");
        return EXIT_FAILURE;
    }

    y = (double *)malloc(n*sizeof(*y));
    if (x == NULL) {
        printf("host malloc(y) failed\n");
        return EXIT_FAILURE;
    }

    for (int i=0; i<n; i++) {
        x[i] =  i+1;
        y[i] = -i-1;
    }

    // DEBUG weirdness on summit where GENE can't see MEMORY_PER_CORE,
    // possibly because the system spectrum mpi uses it in some way.
    if (world_rank == 0) {
        mb_per_core = getenv("MEMORY_PER_CORE");
        if (mb_per_core == NULL) {
            printf("MEMORY_PER_CORE is not set\n");
        } else {
            printf("MEMORY_PER_CORE=%s\n", mb_per_core);
        }
    }

    set_rank_device(world_size, world_rank);
    //CHECK("setDevice", cudaSetDevice(0));

    CHECK( "cublas", cublasCreate(&handle) );

    CHECK( "d_x", cudaMalloc((void**)&d_x, n*sizeof(*d_x)) );
    CHECK( "d_y", cudaMalloc((void**)&d_y, n*sizeof(*d_y)) );

    CHECK( "m_x", cudaMallocManaged((void**)&m_x, n*sizeof(*m_x)) );
    CHECK( "m_y", cudaMallocManaged((void**)&m_y, n*sizeof(*m_y)) );

    CHECK("d_x = x",
          cudaMemcpy(d_x, x, n*sizeof(*x), cudaMemcpyHostToDevice) );
    CHECK("d_y = y",
          cudaMemcpy(d_y, y, n*sizeof(*y), cudaMemcpyHostToDevice) );

    CHECK("m_x = x",
          cudaMemcpy(m_x, x, n*sizeof(*x), cudaMemcpyHostToDevice) );
    CHECK("m_y = y",
          cudaMemcpy(m_y, y, n*sizeof(*y), cudaMemcpyHostToDevice) );

    MEMINFO("d_x", d_x, sizeof(d_x));
    MEMINFO("d_y", d_y, sizeof(d_y));

    MEMINFO("m_x", m_x, sizeof(m_x));
    MEMINFO("m_y", m_y, sizeof(m_y));

    MEMINFO("x", x, sizeof(x));
    MEMINFO("y", y, sizeof(y));

    CHECK("daxpy",
          cublasDaxpy(handle, n, &a, m_x, 1, m_y, 1) );

    CHECK("daxpy sync", cudaDeviceSynchronize());
    
    /*
    CHECK("y = d_y",
          cudaMemcpy(y, m_y, n*sizeof(*y), cudaMemcpyDeviceToHost) );
    */

    CHECK("y = d_y sync", cudaDeviceSynchronize() );

    sum = 0.0;
    for (int i=0; i<n; i++) {
        //printf("%f\n", y[i]);
        sum += m_y[i];
    }
    printf("%d/%d SUM = %f\n", world_rank, world_size, sum);

    // cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(m_x);
    cudaFree(m_y);
    cublasDestroy(handle);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
