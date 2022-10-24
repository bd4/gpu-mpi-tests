/*
 * =====================================================================================
 *
 *       Filename:  mpi_daxpy_gt.c
 *
 *    Description:  Port to gtensor / gt-blas
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

#include "gtensor/gtensor.h"
#include "gt-blas/blas.h"

void set_rank_device(int n_ranks, int rank) {
    int n_devices, device, ranks_per_device;

    n_devices = gt::backend::clib::device_get_count();

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

    gt::backend::clib::device_set(device);
}


int main(int argc, char **argv) {
    int n = 1024;
    int world_size, world_rank, device_id;
    uint32_t vendor_id;

    double a = 2.0;
    double sum = 0.0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    set_rank_device(world_size, world_rank);

    auto x = gt::empty<double>({n});
    auto y = gt::empty<double>({n});
    auto d_x = gt::empty_device<double>({n});
    auto d_y = gt::empty_device<double>({n});

    for (int i=0; i<n; i++) {
        x[i] =  i+1;
        y[i] = -i-1;
    }

    device_id = gt::backend::clib::device_get();
    vendor_id = gt::backend::clib::device_get_vendor_id(device_id);

    gt::blas::handle_t h;

    gt::copy(x, d_x);
    gt::copy(y, d_y);

    gt::blas::axpy(h, a, d_x, d_y);

    gt::synchronize();

    gt::copy(d_y, y);
    
    sum = 0.0;
    for (int i=0; i<n; i++) {
        //printf("%f\n", y[i]);
        sum += y[i];
    }
    printf("%d/%d [%d:0x%08x] SUM = %f\n", world_rank, world_size, device_id, vendor_id, sum);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
