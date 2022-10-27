/*
 * Test GPU aware MPI on different platforms using a distributed
 * 1d stencil on a 2d array. The exchange in second (non-contiguous)
 * direction forces use of staging buffers, which replicates what
 * is needed for all but the innermost dimension exchanges in the
 * GENE fusion code.
 *
 * Takes optional command line arg for size of each dimension of the domain
 * n_global, in 1024 increments. Default is 8 * 1024 (so 256K plus ghost points
 * in size for doulbles per array), which should fit on any system but may not
 * be enough to tax larger HPC GPUs and MPI impelmentations.
 *
 * There will be four exchange buffers of size 2 * n_global, i.e. 128K each
 * by default.
 */

#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sycl/sycl.hpp"

static constexpr double stencil5[] = {1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0,
                                      -1.0 / 12.0};

constexpr int idx2(int n, int row, int col)
{
  return row + col * n;
}

/*
 * Calculate 1d stencil of second dimension of 2d array on GPU. Out array must
 * be contiguous column major nrows x ncols array, while in array must be
 * (nrows)x(ncols+4) to accomodate 2 ghost points in each direction for the
 * second dimension.
 *
 * Returns sycl event, async with respect to host.
 */
auto stencil2d_1d_5(sycl::queue& q, int nrows, int ncols, double* out2d,
                    const double* in2d, double scale)
{
  // Note: swap index order; SYCL is row-major oriented, and this example
  // is col-major
  auto range = sycl::range<2>(ncols, nrows);
  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range, [=](sycl::item<2> item) {
      int row = item.get_id(1);
      int col = item.get_id(0);
      int base_idx = idx2(nrows, row, col);
      int stride = nrows;
      out2d[base_idx] = (stencil5[0] * in2d[base_idx + 0 * stride] +
                         stencil5[1] * in2d[base_idx + 1 * stride] +
                         stencil5[2] * in2d[base_idx + 2 * stride] +
                         stencil5[3] * in2d[base_idx + 3 * stride] +
                         stencil5[4] * in2d[base_idx + 4 * stride]) *
                        scale;
    });
  });
  return e;
}

/*
 * Copy slice of second (non-contiguous) dimension of in array into contiguous
 * buffer out. In has dimension nrows x ncols, buf has dimension nrows x (end
 * -start + 1).
 */
auto buf_from_view(sycl::queue& q, int nrows, double* buf, double* in,
                   int start, int end)
{
  // Note: reverse index order because SYCL is row-major
  auto range = sycl::range<2>(end - start, nrows);
  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range, [=](sycl::item<2> item) {
      int row = item.get_id(1);
      int col = item.get_id(0);
      buf[idx2(nrows, row, col)] = in[idx2(nrows, row, start + col)];
    });
  });
  return e;
}

/*
 * Copy contiguous buffer into second (non-contiguous) dimension of array as a
 * slice. Out has dimension nrows x ncols, buf has dimension nrows x (end -
 * start + 1). ncols is not actually needed or used, just need start to end
 * to be valid indexes.
 */
auto buf_to_view(sycl::queue& q, int nrows, double* out, double* buf, int start,
                 int end)
{
  // Note: reverse index order because SYCL is row-major
  auto range = sycl::range<2>(end - start, nrows);
  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range, [=](sycl::item<2> item) {
      int row = item.get_id(1);
      int col = item.get_id(0);
      out[idx2(nrows, row, start + col)] = buf[idx2(nrows, row, col)];
    });
  });
  return e;
}

void test_buf_view(sycl::queue& q, const int n)
{
  double* data = sycl::malloc_host<double>(n * n, q);
  double* buf = sycl::malloc_host<double>(n * 2, q);
  double* buf2 = sycl::malloc_host<double>(n * 2, q);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      data[j + i * n] = i + j / 1000.0;
    }
    buf2[i] = 100.0 + i;
    buf2[i + n] = 100.0 + i + 0.1;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("data[%d, %d] = %f\n", j, i, data[j + i * n]);
    }
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < n; j++) {
      printf("buf2[%d, %d] = %f\n", j, i, buf2[j + i * n]);
    }
  }

  buf_from_view(q, n, buf, data, 0, 2).wait();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < n; j++) {
      printf("buf[%d, %d] = %f\n", j, i, buf[j + i * n]);
    }
  }

  buf_to_view(q, n, data, buf2, n - 2, n).wait();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("data[%d, %d] = %f\n", j, i, data[j + i * n]);
    }
  }
}

/*
 * Calculate the norm of the difference of two arrays, as sqrt of sum of
 * squared distances.
 */
double diff_norm(sycl::queue& q, int size, double* d_a, double* d_b)
{
  double result = 0.0;
  sycl::buffer<double> result_buf{&result, 1};
  {
    sycl::range<1> range(size);
    auto e = q.submit([&](sycl::handler& cgh) {
      auto reducer = sycl::reduction(result_buf, cgh, 0.0, std::plus<>{});
      cgh.parallel_for(range, reducer, [=](sycl::id<1> idx, auto& r) {
        double diff = d_a[idx] - d_b[idx];
        r.combine(diff * diff);
      });
    });
    e.wait();
  }
  return std::sqrt(result_buf.get_host_access()[0]);
}

sycl::queue get_rank_queue(int n_ranks, int rank)
{
  int n_devices, device_idx, ranks_per_device;

  cl::sycl::platform p{cl::sycl::default_selector()};
  auto devices = p.get_devices();
  n_devices = devices.size();

  if (n_ranks > n_devices) {
    if (n_ranks % n_devices != 0) {
      printf(
        "ERROR: Number of ranks (%d) not a multiple of number of GPUs (%d)\n",
        n_ranks, n_devices);
      exit(EXIT_FAILURE);
    }
    ranks_per_device = n_ranks / n_devices;
    device_idx = rank / ranks_per_device;
  } else {
    ranks_per_device = 1;
    device_idx = rank;
  }

  // printf("n_devices = %d\n", n_devices);
  // printf("device_idx = %d\n", device_idx);

  return sycl::queue{devices[device_idx],
                     cl::sycl::property::queue::in_order()};
}

// exchange in non-contiguous second dimension, staging into contiguous buffers
// on device
void boundary_exchange_y(MPI_Comm comm, int world_size, int rank,
                         sycl::queue& q, int n_global, int n_local, int n_bnd,
                         double* d_z, bool stage_host = false)
{
  int buf_size = n_global * n_bnd;
  static double* sbuf_l = nullptr;
  static double* sbuf_r = nullptr;
  static double* rbuf_l = nullptr;
  static double* rbuf_r = nullptr;

  if (sbuf_l == nullptr) {
    sbuf_l = sycl::malloc_device<double>(buf_size, q);
    sbuf_r = sycl::malloc_device<double>(buf_size, q);
    rbuf_l = sycl::malloc_device<double>(buf_size, q);
    rbuf_r = sycl::malloc_device<double>(buf_size, q);
  }

  static double* h_sbuf_l = nullptr;
  static double* h_sbuf_r = nullptr;
  static double* h_rbuf_l = nullptr;
  static double* h_rbuf_r = nullptr;
  if (stage_host && h_sbuf_l == nullptr) {
    h_sbuf_l = sycl::malloc_host<double>(buf_size, q);
    h_sbuf_r = sycl::malloc_host<double>(buf_size, q);
    h_rbuf_l = sycl::malloc_host<double>(buf_size, q);
    h_rbuf_r = sycl::malloc_host<double>(buf_size, q);
  }

  MPI_Request req_l[2];
  MPI_Request req_r[2];

  int rank_l = rank - 1;
  int rank_r = rank + 1;

  // start async copy of ghost points into send buffers
  if (rank_l >= 0) {
    // printf("rank_l = %d\n", rank_l); fflush(nullptr);
    // sbuf_l = d_z.view(_all, _s(n_bnd, 2 * n_bnd));
    auto e = buf_from_view(q, n_global, sbuf_l, d_z, n_bnd, 2 * n_bnd);
    if (stage_host) {
      q.copy(sbuf_l, h_sbuf_l, buf_size, e);
      /*
      for (int i = 0; i < n_bnd; i++) {
        for (int j = 0; j < n_global; j++) {
          int idx = idx2(n_global, j, i);
          printf("sbuf_l[%d, %d] = %f\n", j, i, h_sbuf_l[idx]);
          fflush(nullptr);
        }
      }
      */
    }
  }
  if (rank_r < world_size) {
    // printf("rank_r = %d\n", rank_r); fflush(nullptr);
    // sbuf_r = d_z.view(_all, _s(-2 * n_bnd, -n_bnd));
    auto e = buf_from_view(q, n_global, sbuf_r, d_z, n_local, n_local + n_bnd);
    if (stage_host) {
      q.copy(sbuf_r, h_sbuf_r, buf_size, e);
      /*
      for (int i = 0; i < n_bnd; i++) {
        for (int j = 0; j < n_global; j++) {
          int idx = idx2(n_global, j, i);
          printf("sbuf_r[%d, %d] = %f\n", j, i, h_sbuf_r[idx]);
          fflush(nullptr);
        }
      }
      */
    }
  }

  // initiate async recv
  if (rank_l >= 0) {
    double* rbuf_l_data = nullptr;
    if (stage_host) {
      rbuf_l_data = h_rbuf_l;
    } else {
      rbuf_l_data = rbuf_l;
    }
    MPI_Irecv(rbuf_l_data, buf_size, MPI_DOUBLE, rank_l, 123, comm, &req_l[0]);
  }

  if (rank_r < world_size) {
    double* rbuf_r_data = nullptr;
    if (stage_host) {
      rbuf_r_data = h_rbuf_r;
    } else {
      rbuf_r_data = rbuf_r;
    }
    MPI_Irecv(rbuf_r_data, buf_size, MPI_DOUBLE, rank_r, 456, comm, &req_r[0]);
  }

  // wait for send buffer fill
  q.wait();

  // initiate async sends
  if (rank_l >= 0) {
    double* sbuf_l_data = nullptr;
    if (stage_host) {
      sbuf_l_data = h_sbuf_l;
    } else {
      sbuf_l_data = sbuf_l;
    }
    MPI_Isend(sbuf_l_data, buf_size, MPI_DOUBLE, rank_l, 456, comm, &req_l[1]);
  }

  if (rank_r < world_size) {
    double* sbuf_r_data = nullptr;
    if (stage_host) {
      sbuf_r_data = h_sbuf_r;
    } else {
      sbuf_r_data = sbuf_r;
    }
    MPI_Isend(sbuf_r_data, buf_size, MPI_DOUBLE, rank_r, 123, comm, &req_r[1]);
  }

  // wait for send/recv to complete, then copy data back into main data array
  int mpi_rval;
  if (rank_l >= 0) {
    mpi_rval = MPI_Waitall(2, req_l, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_l error: %d\n", mpi_rval);
    }
    if (stage_host) {
      /*
      for (int i = 0; i < n_bnd; i++) {
        for (int j = 0; j < n_global; j++) {
          int idx = idx2(n_global, j, i);
          printf("rbuf_l[%d, %d] = %f\n", j, i, h_rbuf_l[idx]);
          fflush(nullptr);
        }
      }
      */
      q.copy(h_rbuf_l, rbuf_l, buf_size);
    }
    // d_z.view(_all, _s(0, n_bnd)) = rbuf_l;
    buf_to_view(q, n_global, d_z, rbuf_l, 0, n_bnd);
  }
  if (rank_r < world_size) {
    mpi_rval = MPI_Waitall(2, req_r, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_r error: %d\n", mpi_rval);
    }
    if (stage_host) {
      /*
      for (int i = 0; i < n_bnd; i++) {
        for (int j = 0; j < n_global; j++) {
          int idx = idx2(n_global, j, i);
          printf("rbuf_r[%d, %d] = %f\n", j, i, h_rbuf_r[idx]);
          fflush(nullptr);
        }
      }
      */
      q.copy(h_rbuf_r, rbuf_r, buf_size);
    }
    // d_z.view(_all, _s(-n_bnd, _)) = rbuf_r;
    buf_to_view(q, n_global, d_z, rbuf_r, n_local + n_bnd, n_local + 2 * n_bnd);
  }

  q.wait();
}

int main(int argc, char** argv)
{
  // sycl::queue q2{};
  // test_buf_view(q2, 6);
  // return EXIT_SUCCESS;

  // Note: domain will be n_global x n_global plus ghost points in one dimension
  int n_global = 8 * 1024;
  bool stage_host = false;
  int n_iter = 100;
  int n_warmup = 5;

  if (argc > 1) {
    n_global = std::atoi(argv[1]) * 1024;
  }
  if (argc > 2) {
    if (argv[2][0] == '1') {
      stage_host = true;
    }
  }
  if (argc > 3) {
    n_iter = std::atoi(argv[3]);
  }

  int n_sten = 5;
  int n_bnd = (n_sten - 1) / 2;
  int world_size, world_rank, device_id;
  uint32_t vendor_id;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (n_global % world_size != 0) {
    printf("%d nmpi (%d) must be divisor of domain size (%d), exiting\n",
           world_rank, world_size, n_global);
    exit(1);
  }

  const int n_local = n_global / world_size;
  const int n_local_with_ghost = n_local + 2 * n_bnd;

  sycl::queue q = get_rank_queue(world_size, world_rank);

  if (world_rank == 0) {
    printf("n procs    = %d\n", world_size);
    printf("rank       = %d\n", world_rank);
    printf("n_global   = %d\n", n_global);
    printf("n_local    = %d\n", n_local);
    printf("n_iter     = %d\n", n_iter);
    printf("n_warmup   = %d\n", n_warmup);
    printf("stage_host = %d\n", stage_host);
  }

  int z_size = n_global * n_local_with_ghost;
  int dzdy_size = n_global * n_local;

  double* h_z = sycl::malloc_host<double>(z_size, q);
  double* d_z = sycl::malloc_device<double>(z_size, q);

  double* h_dzdy_numeric = sycl::malloc_host<double>(dzdy_size, q);
  double* h_dzdy_actual = sycl::malloc_host<double>(dzdy_size, q);
  double* d_dzdy_numeric = sycl::malloc_device<double>(dzdy_size, q);

  double lx = 8;
  double dx = lx / n_global;
  double lx_local = lx / world_size;
  double scale = n_global / lx;
  auto fn = [](double x, double y) { return x * x + y * y; };
  auto fn_dzdy = [](double x, double y) { return 2 * x; };

  struct timespec start, end;
  double iter_time = 0.0;
  double total_time = 0.0;

  double x_start = world_rank * lx_local;
  for (int i = 0; i < n_local; i++) {
    double xtmp = x_start + i * dx;
    for (int j = 0; j < n_global; j++) {
      double ytmp = j * dx;
      h_z[idx2(n_global, j, i + n_bnd)] = fn(xtmp, ytmp);
      h_dzdy_actual[idx2(n_global, j, i)] = fn_dzdy(xtmp, ytmp);
    }
  }

  // fill boundary points on ends
  if (world_rank == 0) {
    for (int i = 0; i < n_bnd; i++) {
      double xtmp = (i - n_bnd) * dx;
      for (int j = 0; j < n_global; j++) {
        double ytmp = j * dx;
        h_z[idx2(n_global, j, i)] = fn(xtmp, ytmp);
      }
    }
  }
  if (world_rank == world_size - 1) {
    for (int i = 0; i < n_bnd; i++) {
      double xtmp = lx + i * dx;
      for (int j = 0; j < n_global; j++) {
        double ytmp = j * dx;
        h_z[idx2(n_global, j, n_bnd + n_local + i)] = fn(xtmp, ytmp);
      }
    }
  }

  /*
  for (int i = 0; i < 5; i++) {
    int idx = idx2(n_global, 1, i);
    printf("%d row1-l %f\n", world_rank, h_z[idx]);
  }
  for (int i = 0; i < 5; i++) {
    int idx = idx2(n_global, 1, n_local_with_ghost - 1 - i);
    printf("%d row1-r %f\n", world_rank, h_z[idx]);
  }
  */

  q.copy(h_z, d_z, z_size);

  for (int i = 0; i < n_warmup + n_iter; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    boundary_exchange_y(MPI_COMM_WORLD, world_size, world_rank, q, n_global,
                        n_local, n_bnd, d_z, stage_host);
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9);

    if (i >= n_warmup) {
      total_time += iter_time;
    }

    // do some calculation, to try to more closely simulate what happens in GENE
    auto e = stencil2d_1d_5(q, n_global, n_local, d_dzdy_numeric, d_z, scale);
    e.wait();
  }
  printf("%d/%d exchange time %0.8f\n", world_rank, world_size,
         total_time / n_iter);

  q.copy(d_dzdy_numeric, h_dzdy_numeric, dzdy_size).wait();

  /*
  for (int i = 0; i < 5; i++) {
    int idx = idx2(n_global, 8, i);
    printf("%d la %f\n%d ln %f\n", world_rank, h_dzdy_actual[idx], world_rank,
           h_dzdy_numeric[idx]);
  }
  for (int i = 0; i < 5; i++) {
    int idx = idx2(n_global, 8, n_local - 1 - i);
    printf("%d ra %f\n%d rn %f\n", world_rank, h_dzdy_actual[idx], world_rank,
           h_dzdy_numeric[idx]);
  }
  */

  double err_norm = diff_norm(q, dzdy_size, h_dzdy_numeric, h_dzdy_actual);

  printf("%d/%d [%d:0x%08x] err_norm = %.8f\n", world_rank, world_size,
         device_id, vendor_id, err_norm);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
