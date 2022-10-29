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
 *
 * Gtensor is used so a single source can be used for all platforms.
 */

#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gtensor/gtensor.h"
#include "gtensor/reductions.h"

using namespace gt::placeholders;

inline void check(const char* fname, int line, int mpi_rval)
{
  if (mpi_rval != MPI_SUCCESS) {
    printf("%s:%d error %d\n", fname, line, mpi_rval);
    exit(2);
  }
}

#define CHECK(x) check(__FILE__, __LINE__, (x))

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
 * first dimension of a 2d array.
 *
 * Size of the result will be size of z with minus 4 in second dimension.
 */
inline auto stencil2d_1d_5(const gt::gtensor_device<double, 2>& z,
                           const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * z.view(_s(0, -4), _all) +
         stencil(1) * z.view(_s(1, -3), _all) +
         stencil(2) * z.view(_s(2, -2), _all) +
         stencil(3) * z.view(_s(3, -1), _all) +
         stencil(4) * z.view(_s(4, _), _all);
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

// exchange in first dimension, staging into contiguous buffers on device
void boundary_exchange_x(MPI_Comm comm, int world_size, int rank,
                         gt::gtensor_device<double, 2>& d_z, int n_bnd,
                         bool stage_host = false)
{
  auto buf_shape = gt::shape(n_bnd, d_z.shape(1));
  gt::gtensor_device<double, 2> sbuf_l(buf_shape);
  gt::gtensor_device<double, 2> sbuf_r(buf_shape);
  gt::gtensor_device<double, 2> rbuf_r(buf_shape);
  gt::gtensor_device<double, 2> rbuf_l(buf_shape);

  gt::shape_type<2> host_buf_shape;
  if (stage_host) {
    host_buf_shape = buf_shape;
  } else {
    host_buf_shape = {0, 0};
  }
  gt::gtensor<double, 2> h_sbuf_l(host_buf_shape);
  gt::gtensor<double, 2> h_sbuf_r(host_buf_shape);
  gt::gtensor<double, 2> h_rbuf_r(host_buf_shape);
  gt::gtensor<double, 2> h_rbuf_l(host_buf_shape);

  MPI_Request req_l[2];
  MPI_Request req_r[2];

  int rank_l = rank - 1;
  int rank_r = rank + 1;

  // start async copy of ghost points into send buffers
  if (rank_l >= 0) {
    sbuf_l = d_z.view(_s(n_bnd, 2 * n_bnd), _all);
    if (stage_host) {
      gt::copy(sbuf_l, h_sbuf_l);
    }
  }
  if (rank_r <= world_size) {
    sbuf_r = d_z.view(_s(-2 * n_bnd, -n_bnd), _all);
    if (stage_host) {
      gt::copy(sbuf_r, h_sbuf_r);
    }
  }

  // initiate async recv
  if (rank_l >= 0) {
    double* rbuf_l_data = nullptr;
    if (stage_host) {
      rbuf_l_data = h_rbuf_l.data();
    } else {
      rbuf_l_data = rbuf_l.data().get();
    }
    CHECK(MPI_Irecv(rbuf_l_data, rbuf_l.size(), MPI_DOUBLE, rank_l, 123, comm,
                    &req_l[0]));
  }

  if (rank_r < world_size) {
    double* rbuf_r_data = nullptr;
    if (stage_host) {
      rbuf_r_data = h_rbuf_r.data();
    } else {
      rbuf_r_data = rbuf_r.data().get();
    }
    CHECK(MPI_Irecv(rbuf_r_data, rbuf_r.size(), MPI_DOUBLE, rank_r, 456, comm,
                    &req_r[0]));
  }

  // wait for send buffer fill
  gt::synchronize();

  // initiate async sends
  if (rank_l >= 0) {
    double* sbuf_l_data = nullptr;
    if (stage_host) {
      sbuf_l_data = h_sbuf_l.data();
    } else {
      sbuf_l_data = sbuf_l.data().get();
    }
    CHECK(MPI_Isend(sbuf_l_data, sbuf_l.size(), MPI_DOUBLE, rank_l, 456, comm,
                    &req_l[1]));
  }

  if (rank_r < world_size) {
    double* sbuf_r_data = nullptr;
    if (stage_host) {
      sbuf_r_data = h_sbuf_r.data();
    } else {
      sbuf_r_data = sbuf_r.data().get();
    }
    CHECK(MPI_Isend(sbuf_r_data, sbuf_r.size(), MPI_DOUBLE, rank_r, 123, comm,
                    &req_r[1]));
  }

  // wait for send/recv to complete, then copy data back into main data array
  int mpi_rval;
  if (rank_l >= 0) {
    MPI_Status status[2];
    mpi_rval = MPI_Waitall(2, req_l, status);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_l error: %d (%d %d)\n", mpi_rval, status[0].MPI_ERROR,
             status[1].MPI_ERROR);
    }
    if (stage_host) {
      gt::copy(h_rbuf_l, rbuf_l);
    }
    d_z.view(_s(0, n_bnd), _all) = rbuf_l;
  }
  if (rank_r < world_size) {
    MPI_Status status[2];
    mpi_rval = MPI_Waitall(2, req_r, status);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_r error: %d (%d %d)\n", mpi_rval, status[0].MPI_ERROR,
             status[1].MPI_ERROR);
    }
    if (stage_host) {
      gt::copy(h_rbuf_r, rbuf_r);
    }
    d_z.view(_s(-n_bnd, _), _all) = rbuf_r;
  }

  gt::synchronize();
}

int main(int argc, char** argv)
{
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

  CHECK(MPI_Init(NULL, NULL));

  CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

  if (n_global % world_size != 0) {
    printf("%d nmpi (%d) must be divisor of domain size (%d), exiting\n",
           world_rank, world_size, n_global);
    exit(1);
  }

  const int n_local = n_global / world_size;
  const int n_local_with_ghost = n_local + 2 * n_bnd;

  set_rank_device(world_size, world_rank);
  device_id = gt::backend::clib::device_get();
  vendor_id = gt::backend::clib::device_get_vendor_id(device_id);

  if (world_rank == 0) {
    printf("n procs    = %d\n", world_size);
    printf("n_global   = %d\n", n_global);
    printf("n_local    = %d\n", n_local);
    printf("n_iter     = %d\n", n_iter);
    printf("n_warmup   = %d\n", n_warmup);
    printf("stage_host = %d\n", stage_host);
  }

  auto h_z = gt::empty<double>({n_local_with_ghost, n_global});
  auto d_z = gt::empty_device<double>({n_local_with_ghost, n_global});

  auto h_dzdx_numeric = gt::empty<double>({n_local, n_global});
  auto h_dzdx_actual = gt::empty<double>({n_local, n_global});
  auto d_dzdx_numeric = gt::empty_device<double>({n_local, n_global});

  double lx = 8;
  double dx = lx / n_global;
  double lx_local = lx / world_size;
  double scale = n_global / lx;
  auto fn = [](double x, double y) { return x * x * x + y * y; };
  auto fn_dzdx = [](double x, double y) { return 3 * x * x; };

  struct timespec start, end;
  double iter_time = 0.0;
  double total_time = 0.0;

  double x_start = world_rank * lx_local;
  for (int j = 0; j < n_global; j++) {
    double ytmp = j * dx;
    for (int i = 0; i < n_local; i++) {
      double xtmp = x_start + i * dx;
      h_z(i + n_bnd, j) = fn(xtmp, ytmp);
      h_dzdx_actual(i, j) = fn_dzdx(xtmp, ytmp);
    }
  }

  // fill boundary points on ends
  if (world_rank == 0) {
    for (int j = 0; j < n_global; j++) {
      double ytmp = j * dx;
      for (int i = 0; i < n_bnd; i++) {
        double xtmp = (i - n_bnd) * dx;
        h_z(i, j) = fn(xtmp, ytmp);
      }
    }
  }
  if (world_rank == world_size - 1) {
    for (int j = 0; j < n_global; j++) {
      double ytmp = j * dx;
      for (int i = 0; i < n_bnd; i++) {
        double xtmp = lx + i * dx;
        h_z(n_bnd + n_local + i, j) = fn(xtmp, ytmp);
      }
    }
  }

  /*
  for (int i = 0; i < 5; i++) {
    printf("%d row1-l %f\n", world_rank, h_z(1, i));
  }
  for (int i = 0; i < 5; i++) {
    printf("%d row1-r %f\n", world_rank, h_z(1, n_local_with_ghost - 1 - i));
  }
  */

  gt::copy(h_z, d_z);
  // gt::synchronize();

  for (int i = 0; i < n_warmup + n_iter; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    boundary_exchange_x(MPI_COMM_WORLD, world_size, world_rank, d_z, n_bnd,
                        stage_host);
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9);

    if (i >= n_warmup) {
      total_time += iter_time;
    }

    // do some calculation, to try to more closely simulate what happens in GENE
    d_dzdx_numeric = stencil2d_1d_5(d_z, stencil5) * scale;
    gt::synchronize();
  }
  printf("%d/%d exchange time %0.8f\n", world_rank, world_size,
         total_time / n_iter);

  gt::copy(d_dzdx_numeric, h_dzdx_numeric);

  /*
  for (int i = 0; i < 5; i++) {
    printf("%d la %f\n%d ln %f\n", world_rank, h_dzdx_actual(8, i), world_rank,
           h_dzdx_numeric(8, i));
  }
  for (int i = 0; i < 5; i++) {
    int idx = n_local - 1 - i;
    printf("%d ra %f\n%d rn %f\n", world_rank, h_dzdx_actual(8, idx),
           world_rank, h_dzdx_numeric(8, idx));
  }
  */

  double err_norm = std::sqrt(gt::sum_squares(h_dzdx_numeric - h_dzdx_actual));

  printf("%d/%d [%d:0x%08x] err_norm = %.8f\n", world_rank, world_size,
         device_id, vendor_id, err_norm);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
