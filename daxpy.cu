/*
 * =====================================================================================
 *
 *       Filename:  daxpy.c
 *
 *    Description:  Test cublas DAXPY, specifically to verify usage on
 *                  summit with GPUMPS and all 6 GPUs shared over 42 procs.
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

#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "cuda_runtime_api.h"

#define GPU_CHECK_CALLS
#include "cuda_error.h"

// column major
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static cublasHandle_t handle;


int main(int argc, char **argv) {
    int n = 1024;

    double a = 2.0;
    double sum = 0.0;

    double *x, *y, *d_x, *d_y;

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
        x[i] = i+1;
        y[i] = -i-1;
    }

    //CHECK("setDevice", cudaSetDevice(0));

    CHECK( "cublas", cublasCreate(&handle) );

    CHECK( "d_x", cudaMalloc((void**)&d_x, n*sizeof(*d_x)) );
    CHECK( "d_y", cudaMalloc((void**)&d_y, n*sizeof(*d_y)) );

    CHECK("d_x = x",
          cudaMemcpy(d_x, x, n*sizeof(*x), cudaMemcpyHostToDevice) );
    CHECK("d_y = y",
          cudaMemcpy(d_y, y, n*sizeof(*y), cudaMemcpyHostToDevice) );

    CHECK("daxpy",
          cublasDaxpy(handle, n, &a, d_x, 1, d_y, 1) );

    CHECK("daxpy sync", cudaDeviceSynchronize());
    
    CHECK("y = d_y",
          cudaMemcpy(y, d_y, n*sizeof(*y), cudaMemcpyDeviceToHost) );

    CHECK("y = d_y sync", cudaDeviceSynchronize() );

    sum = 0.0;
    for (int i=0; i<n; i++) {
        printf("%f\n", y[i]);
        sum += y[i];
    }
    printf("SUM = %f\n", sum);

    // cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    return EXIT_SUCCESS;
}
