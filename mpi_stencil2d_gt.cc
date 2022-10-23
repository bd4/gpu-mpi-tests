/*
 * Test GPU aware MPI on different platforms using a simple
 * distributed 1d stencil as an example. Gtensor is used so
 * a single source can be used for all platforms.
 */

#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gtensor/gtensor.h"
#include "gtensor/reductions.h"

using namespace gt::placeholders;

// little hack to make code parameterizable on managed vs device memory
namespace gt
{

namespace ext
{
namespace detail
{

template <typename T, gt::size_type N, typename S = gt::space::device>
struct gthelper
{
  using gtensor = gt::gtensor<T, N, S>;
};

#ifdef GTENSOR_HAVE_DEVICE

template <typename T, gt::size_type N>
struct gthelper<T, N, gt::space::managed>
{
  using gtensor = gt::gtensor_container<gt::space::managed_vector<T>, N>;
};
#endif

} // namespace detail

template <typename T, gt::size_type N, typename S = gt::space::device>
using gtensor2 = typename detail::gthelper<T, N, S>::gtensor;

} // namespace ext

} // namespace gt

static const gt::gtensor<double, 1> stencil5 = {1.0 / 12.0, -2.0 / 3.0, 0.0,
                                                2.0 / 3.0, -1.0 / 12.0};

/*
 * Return unevaluated expression that calculates the 1d stencil in the
 * second dimension of a 2d array.
 *
 * Size of the result will be size of z with minus 4 in second dimension.
 */
inline auto stencil2d_1d_5(const gt::gtensor_device<double, 2>& z,
                           const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * z.view(_all, _s(0, -4)) +
         stencil(1) * z.view(_all, _s(1, -3)) +
         stencil(2) * z.view(_all, _s(2, -2)) +
         stencil(3) * z.view(_all, _s(3, -1)) +
         stencil(4) * z.view(_all, _s(4, _));
}

void set_rank_device(int n_ranks, int rank)
{
  int n_devices, device, ranks_per_device;

  n_devices = gt::backend::clib::device_get_count();

  if (n_ranks > n_devices) {
    if (n_ranks % n_devices != 0) {
      printf(
        "ERROR: Number of ranks (%d) not a multiple of number of GPUs (%d)\n",
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

// exchange in non-contiguous second dimension, staging into contiguous buffers
// on device
void boundary_exchange_y(MPI_Comm comm, int world_size, int rank,
                         gt::gtensor_device<double, 2>& d_z, int n_bnd)
{
  auto buf_shape = gt::shape(d_z.shape(0), n_bnd);
  gt::gtensor_device<double, 2> sbuf_l(buf_shape);
  gt::gtensor_device<double, 2> sbuf_r(buf_shape);
  gt::gtensor_device<double, 2> rbuf_r(buf_shape);
  gt::gtensor_device<double, 2> rbuf_l(buf_shape);

  MPI_Request req_l[2];
  MPI_Request req_r[2];

  int rank_l = rank - 1;
  int rank_r = rank + 1;

  // start async copy of ghost points into send buffers
  if (rank_l >= 0) {
    sbuf_l = d_z.view(_all, _s(n_bnd, 2 * n_bnd));
  }
  if (rank_r <= world_size) {
    sbuf_r = d_z.view(_all, _s(-2 * n_bnd, -n_bnd));
  }

  // initiate async recv
  if (rank_l >= 0) {
    // send/recv left boundary
    MPI_Irecv(gt::raw_pointer_cast(rbuf_l.data()), n_bnd, MPI_DOUBLE, rank_l,
              123, comm, &req_l[0]);
  }

  if (rank_r < world_size) {
    // send/recv right boundary
    MPI_Irecv(gt::raw_pointer_cast(rbuf_r.data()), n_bnd, MPI_DOUBLE, rank_r,
              456, comm, &req_r[0]);
  }

  // wait for send buffer fill
  gt::synchronize();

  // initiate async sends
  if (rank_l >= 0) {
    MPI_Isend(gt::raw_pointer_cast(sbuf_l.data()), n_bnd, MPI_DOUBLE, rank_l,
              456, comm, &req_l[1]);
  }

  if (rank_r < world_size) {
    MPI_Isend(gt::raw_pointer_cast(sbuf_r.data()), n_bnd, MPI_DOUBLE, rank_r,
              123, comm, &req_r[1]);
  }

  int mpi_rval;
  if (rank_l >= 0) {
    mpi_rval = MPI_Waitall(2, req_l, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_l error: %d\n", mpi_rval);
    }
  }
  if (rank_r < world_size) {
    mpi_rval = MPI_Waitall(2, req_r, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_r error: %d\n", mpi_rval);
    }
  }

  // copy recv data into non-contiguous location
  if (rank_l >= 0) {
    d_z.view(_all, _s(0, n_bnd)) = rbuf_l;
  }
  if (rank_r <= world_size) {
    d_z.view(_all, _s(-n_bnd, _)) = rbuf_r;
  }

  gt::synchronize();
}

int main(int argc, char** argv)
{
  int n_global = 8 * 1024;

  if (argc > 1) {
    n_global = std::atoi(argv[1]) * 1024;
  }

  int n_sten = 5;
  int n_bnd = (n_sten - 1) / 2;
  int world_size, world_rank, device_id;
  uint32_t vendor_id;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const int n_local = n_global / world_size;
  const int n_local_with_ghost = n_local + 2 * n_bnd;

  set_rank_device(world_size, world_rank);
  device_id = gt::backend::clib::device_get();
  vendor_id = gt::backend::clib::device_get_vendor_id(device_id);

  if (world_rank == 0) {
    printf("n procs  = %d\n", world_size);
    printf("n_global = %d\n", n_global);
    printf("n_local  = %d\n", n_local);
  }

  auto h_z = gt::empty<double>({n_global, n_local_with_ghost});
  auto d_z = gt::empty_device<double>({n_global, n_local_with_ghost});

  auto h_dzdy_numeric = gt::empty<double>({n_global, n_local});
  auto h_dzdy_actual = gt::empty<double>({n_global, n_local});
  auto d_dzdy_numeric = gt::empty_device<double>({n_global, n_local});

  double lx = 8;
  double dx = lx / n_global;
  double lx_local = lx / world_size;
  double scale = n_global / lx;
  auto fn = [](double x, double y) { return x * x + y * y; };
  auto fn_dzdy = [](double x, double y) { return 2 * x; };

  struct timespec start, end;
  double seconds = 0.0;

  double x_start = world_rank * lx_local;
  for (int i = 0; i < n_local; i++) {
    double xtmp = x_start + i * dx;
    for (int j = 0; j < n_global; j++) {
      double ytmp = j * dx;
      h_z(j, i + n_bnd) = fn(xtmp, ytmp);
      h_dzdy_actual(j, i) = fn_dzdy(xtmp, ytmp);
    }
  }

  // fill boundary points on ends
  if (world_rank == 1) {
    for (int i = 0; i < n_bnd; i++) {
      double xtmp = (i - n_bnd) * dx;
      for (int j = 0; j < n_global; j++) {
        double ytmp = j * dx;
        h_z(j, i + n_bnd) = fn(xtmp, ytmp);
      }
    }
  }
  if (world_rank == world_size - 1) {
    for (int i = 0; i < n_bnd; i++) {
      double xtmp = lx + i * dx;
      for (int j = 0; j < n_global; j++) {
        double ytmp = j * dx;
        h_z(j, n_bnd + n_local + i) = fn(xtmp, ytmp);
      }
    }
  }

  gt::copy(h_z, d_z);
  // gt::synchronize();

  clock_gettime(CLOCK_MONOTONIC, &start);
  boundary_exchange_y(MPI_COMM_WORLD, world_size, world_rank, d_z, n_bnd);
  // gt::synchronize();
  clock_gettime(CLOCK_MONOTONIC, &end);
  seconds =
    ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9);
  printf("%d/%d exchange time %0.4f\n", world_rank, world_size, seconds);

  d_dzdy_numeric = stencil2d_1d_5(d_z, stencil5) * scale;
  // gt::synchronize();

  gt::copy(d_dzdy_numeric, h_dzdy_numeric);
  // gt::synchronize();

  /*
  for (int i = 0; i < 5; i++) {
    printf("{0} l {1}\n{0} l {2}\n", world_rank, h_dzdy_actual(i),
               h_dzdy_numeric(i));
  }
  for (int i = 0; i < 5; i++) {
    int idx = n_local - 1 - i;
    printf("{0} r {1}\n{0} r {2}\n", world_rank, h_dzdy_actual(idx),
               h_dzdy_numeric(idx));
  }
  */

  double err_norm = std::sqrt(gt::sum_squares(h_dzdy_numeric - h_dzdy_actual));

  printf("%d/%d [%d:0x%08x] err_norm = %.8f\n", world_rank, world_size,
         device_id, vendor_id, err_norm);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
