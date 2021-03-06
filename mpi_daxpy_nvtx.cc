/*
 * ===========================================================================
 *
 *       Filename:  mpi_daxpy_nvtx.c
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
 * ===========================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "cuda_runtime_api.h"

#include "nvToolsExt.h"
#include "cuda_profiler_api.h"

#define GPU_CHECK_CALLS
#include "cuda_error.h"

// column major
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//#define BARRIER


static cublasHandle_t handle;

static const int MB = 1024*1024;


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


int get_node_count(int n_ranks) {
    int shm_size;
    MPI_Comm shm_comm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_size);

    MPI_Comm_free(&shm_comm);
    return n_ranks / shm_size;
}


int main(int argc, char **argv) {
    const int n_per_node = 48*MB;
    int nodes = 1;
    int nall = n_per_node;
    int n = 0;
    int world_size, world_rank;

    size_t free_mem, total_mem;

    double a = 2.0;
    double sum = 0.0;

    double start_time = 0.0;
    double end_time = 0.0;
    double k_start_time = 0.0;
    double k_end_time = 0.0;
    double g_start_time = 0.0;
    double g_end_time = 0.0;
    double b_start_time = 0.0;
    double b_end_time = 0.0;

#ifndef MANAGED
    double *h_x, *h_y;
    double *h_allx, *h_ally;
#endif

    double *d_x, *d_y;
    double *d_allx, *d_ally;

    char *mb_per_core;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    nodes = get_node_count(world_size);

    // hack: assume max 6 mpi per node, so we use bigger
    // arrays on multi-node runs
    /*
    if (world_size > 6) {
        nodes = (world_size + 5) / 6;
    }
    */

    nall = nodes * n_per_node;
    n = nall / world_size;

    if (world_rank == 0) {
        printf("%d nodes, %d ranks, %d elements each, total %d\n",
               nodes, world_size, n, nall);
    }

    /*
    x = (double *)malloc(n*sizeof(*x));
    if (x == NULL) {
        printf("host malloc(x) failed\n");
        return EXIT_FAILURE;
    }

    y = (double *)malloc(n*sizeof(*y));
    if (y == NULL) {
        printf("host malloc(y) failed\n");
        return EXIT_FAILURE;
    }
    */

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

    cudaProfilerStart();
    start_time = MPI_Wtime();

    CHECK( "cublas", cublasCreate(&handle) );

    /*
    CHECK( "d_x", cudaMalloc((void**)&d_x, n*sizeof(*d_x)) );
    CHECK( "d_y", cudaMalloc((void**)&d_y, n*sizeof(*d_y)) );
    */

    nvtxRangePushA("allocateArrays");
#ifdef MANAGED
    CHECK( "d_x", cudaMallocManaged((void**)&d_x, n*sizeof(*d_x)) );
    CHECK( "d_y", cudaMallocManaged((void**)&d_y, n*sizeof(*d_y)) );
    CHECK( "d_allx", cudaMallocManaged((void**)&d_allx,
           n*sizeof(*d_allx)*world_size) );
    CHECK( "d_ally", cudaMallocManaged((void**)&d_ally,
           n*sizeof(*d_ally)*world_size) );
#else
    CHECK( "h_x", cudaMallocHost((void**)&h_x, n*sizeof(*h_x)) );
    CHECK( "h_y", cudaMallocHost((void**)&h_y, n*sizeof(*h_y)) );
    CHECK( "d_x", cudaMalloc((void**)&d_x, n*sizeof(*d_x)) );
    CHECK( "d_y", cudaMalloc((void**)&d_y, n*sizeof(*d_y)) );
    CHECK( "d_allx", cudaMalloc((void**)&d_allx,
           n*sizeof(*d_allx)*world_size) );
    CHECK( "d_ally", cudaMalloc((void**)&d_ally,
           n*sizeof(*d_ally)*world_size) );
    CHECK( "h_allx", cudaMallocHost((void**)&h_allx,
           n*sizeof(*h_allx)*world_size) );
    CHECK( "h_ally", cudaMallocHost((void**)&h_ally,
           n*sizeof(*h_ally)*world_size) );
#endif
    nvtxRangePop();

    if (world_rank == 0) {
        CHECK( "memInfo", cudaMemGetInfo(&free_mem, &total_mem) );
        printf("GPU memory %0.3f / %0.3f (%0.3f) MB\n", free_mem/(double)MB,
               (double)total_mem/MB, (double)(total_mem-free_mem)/MB);
    }

    nvtxRangePushA("initializeArrays");
#ifdef MANAGED
    for (int i=0; i<n; i++) {
        d_x[i] =   (i+1)/(double)n;
        d_y[i] =  -d_x[i];
    }
#else
    for (int i=0; i<n; i++) {
        h_x[i] =   (i+1)/(double)n;
        h_y[i] =  -h_x[i];
    }
    nvtxRangePushA("copyInput");
    CHECK("d_x = h_x",
          cudaMemcpy(d_x, h_x, n*sizeof(*h_x), cudaMemcpyHostToDevice) );
    CHECK("d_y = h_y",
          cudaMemcpy(d_y, h_y, n*sizeof(*h_y), cudaMemcpyHostToDevice) );
    nvtxRangePop();
#endif
    nvtxRangePop();

    //MEMINFO("d_x", d_x, sizeof(d_x));
    //MEMINFO("d_y", d_y, sizeof(d_y));
    //MEMINFO("x", x, sizeof(x));
    //MEMINFO("y", y, sizeof(y));

    MEMINFO("d_x", d_x, sizeof(d_x));
    MEMINFO("d_y", d_y, sizeof(d_y));

#ifndef MANAGED
    MEMINFO("h_x", h_x, sizeof(h_x));
    MEMINFO("h_y", h_y, sizeof(h_y));
    MEMINFO("h_allx", h_allx, sizeof(h_allx));
    MEMINFO("h_ally", h_ally, sizeof(h_ally));
#endif

    k_start_time = MPI_Wtime();
    nvtxRangePushA("cublasDaxpy");
    CHECK("daxpy",
          cublasDaxpy(handle, n, &a, d_x, 1, d_y, 1) );

    CHECK("daxpy sync", cudaDeviceSynchronize());
    nvtxRangePop();
    k_end_time = MPI_Wtime();
    
    nvtxRangePushA("localSum");
#ifdef MANAGED
    sum = 0.0;
    for (int i=0; i<n; i++) {
        sum += d_y[i];
    }
#else
    nvtxRangePushA("copyOutput");
    CHECK("h_y = d_y",
          cudaMemcpy(h_y, d_y, n*sizeof(*h_y), cudaMemcpyDeviceToHost) );
    nvtxRangePop();
    sum = 0.0;
    for (int i=0; i<n; i++) {
        sum += h_y[i];
    }
#endif
    nvtxRangePop();
    printf("%d/%d SUM = %f\n", world_rank, world_size, sum);

    nvtxRangePushA("copyPrepAllxInplace");
    cudaMemcpy(d_allx+(world_rank*n), d_x, n*sizeof(*d_x), cudaMemcpyDeviceToDevice);
    nvtxRangePop();

#ifdef BARRIER
    b_start_time = MPI_Wtime();
    nvtxRangePushA("mpiBarrier");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop();
    b_end_time = MPI_Wtime();
#endif

    g_start_time = MPI_Wtime();
    nvtxRangePushA("mpiAllGather");
    nvtxRangePushA("x");
    MPI_Allgather(MPI_IN_PLACE, n, MPI_DOUBLE, d_allx, n, MPI_DOUBLE, MPI_COMM_WORLD);
    nvtxRangePop();
    nvtxRangePushA("y");
    MPI_Allgather(d_y, n, MPI_DOUBLE, d_ally, n, MPI_DOUBLE, MPI_COMM_WORLD);
    nvtxRangePop();
    nvtxRangePop();
    g_end_time = MPI_Wtime();

    sum = 0.0;
    nvtxRangePushA("allSum");
#ifdef MANAGED
    for (int i=0; i<n*world_size; i++) {
        sum += d_ally[i];
    }
#else
    nvtxRangePushA("copyAlly");
    CHECK("h_ally = d_ally",
          cudaMemcpy(h_ally, d_ally, n*sizeof(*h_ally)*world_size,
                     cudaMemcpyDeviceToHost) );
    nvtxRangePop();
    for (int i=0; i<n*world_size; i++) {
        sum += h_ally[i];
    }
#endif
    nvtxRangePop();
    printf("%d/%d ALLSUM = %f\n", world_rank, world_size, sum);

    // cleanup
    nvtxRangePushA("free");
#ifndef MANAGED
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_allx);
    cudaFreeHost(h_ally);
#endif
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_allx);
    cudaFree(d_ally);

    nvtxRangePop();

    end_time = MPI_Wtime();
    cudaProfilerStop();

    cublasDestroy(handle);
    MPI_Finalize();

    printf("%d/%d TIME total  : %0.3f\n", world_rank, world_size,
           end_time-start_time);
    printf("%d/%d TIME kernel : %0.3f\n", world_rank, world_size,
           k_end_time-k_start_time);
    printf("%d/%d TIME barrier: %0.3f\n", world_rank, world_size,
           b_end_time-b_start_time);
    printf("%d/%d TIME gather : %0.3f\n", world_rank, world_size,
           g_end_time-g_start_time);

    return EXIT_SUCCESS;
}
