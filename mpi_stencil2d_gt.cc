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
template <typename S>
inline auto stencil2d_1d_5_d0(const gt::ext::gtensor2<double, 2, S>& z,
                              const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * z.view(_s(0, -4), _all) +
         stencil(1) * z.view(_s(1, -3), _all) +
         stencil(2) * z.view(_s(2, -2), _all) +
         stencil(3) * z.view(_s(3, -1), _all) +
         stencil(4) * z.view(_s(4, _), _all);
}

/*
 * Return unevaluated expression that calculates the 1d stencil in the
 * second dimension of a 2d array.
 *
 * Size of the result will be size of z with minus 4 in second dimension.
 */
template <typename S>
inline auto stencil2d_1d_5_d1(const gt::ext::gtensor2<double, 2, S>& z,
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

// exchange in first dimension, staging into contiguous buffers on device
template <typename S>
void boundary_exchange_x(MPI_Comm comm, int world_size, int rank,
                         gt::ext::gtensor2<double, 2, S>& d_z, int n_bnd,
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

// exchange in second dimension, optional staging into device buffer
template <typename S>
void boundary_exchange_y(MPI_Comm comm, int world_size, int rank,
                         gt::ext::gtensor2<double, 2, S>& d_z, int n_bnd,
                         bool stage_device)
{
  gt::shape_type<2> buf_shape;
  if (stage_device) {
    buf_shape = gt::shape(d_z.shape(0), n_bnd);
  } else {
    buf_shape = {0, 0};
  }

  gt::gtensor_device<double, 2> sbuf_l(buf_shape);
  gt::gtensor_device<double, 2> sbuf_r(buf_shape);
  gt::gtensor_device<double, 2> rbuf_r(buf_shape);
  gt::gtensor_device<double, 2> rbuf_l(buf_shape);

  MPI_Request req_l[2];
  MPI_Request req_r[2];

  int rank_l = rank - 1;
  int rank_r = rank + 1;

  auto sv_l = gt::view_strided(d_z, _all, _s(n_bnd, 2 * n_bnd));
  auto sv_r = gt::view_strided(d_z, _all, _s(-2 * n_bnd, -n_bnd));
  auto rv_l = gt::view_strided(d_z, _all, _s(0, n_bnd));
  auto rv_r = gt::view_strided(d_z, _all, _s(-n_bnd, _));

  // start async copy of ghost points into send buffers
  if (stage_device) {
    if (rank_l >= 0) {
      sbuf_l = sv_l;
    }
    if (rank_r <= world_size) {
      sbuf_r = sv_r;
    }
  }

  // initiate async recv
  if (rank_l >= 0) {
    double* rbuf_l_data = nullptr;
    if (stage_device) {
      rbuf_l_data = rbuf_l.data().get();
    } else {
      rbuf_l_data = rv_l.data().get();
    }
    CHECK(MPI_Irecv(rbuf_l_data, rbuf_l.size(), MPI_DOUBLE, rank_l, 123, comm,
                    &req_l[0]));
  }

  if (rank_r < world_size) {
    double* rbuf_r_data = nullptr;
    if (stage_device) {
      rbuf_r_data = rbuf_r.data().get();
    } else {
      rbuf_r_data = rv_r.data().get();
    }
    CHECK(MPI_Irecv(rbuf_r_data, rbuf_r.size(), MPI_DOUBLE, rank_r, 456, comm,
                    &req_r[0]));
  }

  // wait for send buffer fill
  if (stage_device) {
    gt::synchronize();
  }

  // initiate async sends
  if (rank_l >= 0) {
    double* sbuf_l_data = nullptr;
    if (stage_device) {
      sbuf_l_data = sbuf_l.data().get();
    } else {
      sbuf_l_data = sv_l.data().get();
    }
    CHECK(MPI_Isend(sbuf_l_data, sbuf_l.size(), MPI_DOUBLE, rank_l, 456, comm,
                    &req_l[1]));
  }

  if (rank_r < world_size) {
    double* sbuf_r_data = nullptr;
    if (stage_device) {
      sbuf_r_data = sbuf_r.data().get();
    } else {
      sbuf_r_data = sv_r.data().get();
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
    if (stage_device) {
      gt::copy(rbuf_l, rv_l);
    }
  }
  if (rank_r < world_size) {
    MPI_Status status[2];
    mpi_rval = MPI_Waitall(2, req_r, status);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_r error: %d (%d %d)\n", mpi_rval, status[0].MPI_ERROR,
             status[1].MPI_ERROR);
    }
    if (stage_device) {
      gt::copy(rbuf_r, rv_r);
    }
  }

  gt::synchronize();
}

template <int Dim, typename S>
void print_test_name(bool use_buffers)
{
  if constexpr (std::is_same<S, gt::space::device>::value) {
    printf("TEST dim:%d, device , buf:%d", Dim, use_buffers);
  } else {
    printf("TEST dim:%d, managed, buf:%d", Dim, use_buffers);
  }
}

template <typename S, int Dim>
void test(int device_id, uint32_t vendor_id, int world_size, int world_rank,
          int n_global, int n_iter, bool use_buffers, int n_warmup = 5)
{
  // Note: domain will be n_global x n_global plus ghost points in one dimension

  int n_sten = 5;
  int n_bnd = (n_sten - 1) / 2;

  const int n_local = n_global / world_size;

  int nx_local, ny_local;
  int nx_local_ghost, ny_local_ghost;
  int nx_bnd, ny_bnd;

  if constexpr (Dim == 0) {
    nx_bnd = n_bnd;
    ny_bnd = 0;
    nx_local = n_local;
    nx_local_ghost = n_local + 2 * n_bnd;
    ny_local = n_global;
    ny_local_ghost = n_global;
  } else {
    nx_bnd = 0;
    ny_bnd = n_bnd;
    nx_local = n_global;
    nx_local_ghost = n_global;
    ny_local = n_local;
    ny_local_ghost = n_local + 2 * n_bnd;
  }

  gt::shape_type<2> z_shape(nx_local_ghost, ny_local_ghost);
  gt::shape_type<2> dz_shape(nx_local, ny_local);

  auto h_z = gt::empty<double>(z_shape);
  gt::ext::gtensor2<double, 2, S> d_z(z_shape);

  auto h_dz_numeric = gt::empty<double>(dz_shape);
  auto h_dz_actual = gt::empty<double>(dz_shape);
  gt::ext::gtensor2<double, 2, S> d_dz_numeric(dz_shape);

  double ln = 8;
  double delta = ln / n_global;
  double ln_local = ln / world_size;
  double scale = n_global / ln;
  auto fn = [](double x, double y) { return x * x * x + y * y; };
  auto fn_dzdx = [](double x, double y) { return 3 * x * x; };
  auto fn_dzdy = [](double x, double y) { return 2 * y; };

  struct timespec start, end;
  double iter_time = 0.0;
  double total_time = 0.0;

  double x_start = 0, y_start = 0;
  if constexpr (Dim == 0) {
    x_start = world_rank * ln_local;
  } else {
    y_start = world_rank * ln_local;
  }
  for (int j = 0; j < ny_local; j++) {
    double ytmp = y_start + j * delta;
    for (int i = 0; i < nx_local; i++) {
      double xtmp = x_start + i * delta;
      h_z(i + nx_bnd, j + ny_bnd) = fn(xtmp, ytmp);
      if constexpr (Dim == 0) {
        h_dz_actual(i, j) = fn_dzdx(xtmp, ytmp);
      } else {
        h_dz_actual(i, j) = fn_dzdy(xtmp, ytmp);
      }
    }
  }

  // fill boundary points on ends
  if constexpr (Dim == 0) {
    if (world_rank == 0) {
      for (int j = 0; j < ny_local; j++) {
        double ytmp = j * delta;
        for (int i = 0; i < nx_bnd; i++) {
          double xtmp = (i - nx_bnd) * delta;
          h_z(i, j) = fn(xtmp, ytmp);
        }
      }
    }
    if (world_rank == world_size - 1) {
      for (int j = 0; j < ny_local; j++) {
        double ytmp = j * delta;
        for (int i = 0; i < nx_bnd; i++) {
          double xtmp = ln + i * delta;
          h_z(nx_bnd + nx_local + i, j) = fn(xtmp, ytmp);
        }
      }
    }
  } else {
    if (world_rank == 0) {
      for (int j = 0; j < ny_bnd; j++) {
        double ytmp = (j - ny_bnd) * delta;
        for (int i = 0; i < nx_local; i++) {
          double xtmp = i * delta;
          h_z(i, j) = fn(xtmp, ytmp);
        }
      }
    }
    if (world_rank == world_size - 1) {
      for (int j = 0; j < ny_bnd; j++) {
        double ytmp = ln + j * delta;
        for (int i = 0; i < nx_local; i++) {
          double xtmp = i * delta;
          h_z(i, ny_bnd + ny_local + j) = fn(xtmp, ytmp);
        }
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
    if constexpr (Dim == 0) {
      boundary_exchange_x<S>(MPI_COMM_WORLD, world_size, world_rank, d_z, n_bnd,
                             use_buffers);
    } else {
      boundary_exchange_y<S>(MPI_COMM_WORLD, world_size, world_rank, d_z, n_bnd,
                             use_buffers);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9);

    if (i >= n_warmup) {
      total_time += iter_time;
    }

    // do some calculation, to try to more closely simulate what happens in GENE
    if constexpr (Dim == 0) {
      d_dz_numeric = stencil2d_1d_5_d0<S>(d_z, stencil5) * scale;
    } else {
      d_dz_numeric = stencil2d_1d_5_d1<S>(d_z, stencil5) * scale;
    }
    gt::synchronize();
  }
#ifdef DEBUG
  printf("%d/%d exchange time %0.8f ms\n", world_rank, world_size,
         total_time / n_iter * 1000);
#endif

  gt::copy(d_dz_numeric, h_dz_numeric);

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

  double err_norm = std::sqrt(gt::sum_squares(h_dz_numeric - h_dz_actual));

#ifdef DEBUG
  printf("%d/%d [%d:0x%08x] err_norm = %.8f\n", world_rank, world_size,
         device_id, vendor_id, err_norm);
#endif

  double time_sum;
  MPI_Reduce(&total_time, &time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double err_sum;
  MPI_Reduce(&err_norm, &err_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    print_test_name<Dim, S>(use_buffers);
    printf("; %0.8f, err=%0.8f\n", time_sum, err_sum);
  }
}

int main(int argc, char** argv)
{
  using S = gt::space::managed;

  // Note: domain will be n_global x n_global plus ghost points in one dimension
  int n_global = 8 * 1024;
  int n_iter = 100;
  int n_warmup = 5;

  if (argc > 1) {
    n_global = std::atoi(argv[1]) * 1024;
  }
  if (argc > 2) {
    n_iter = std::atoi(argv[2]);
  }

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

  set_rank_device(world_size, world_rank);
  device_id = gt::backend::clib::device_get();
  vendor_id = gt::backend::clib::device_get_vendor_id(device_id);

  if (world_rank == 0) {
    printf("n procs    = %d\n", world_size);
    printf("n_global   = %d\n", n_global);
    printf("n_local    = %d\n", n_local);
    printf("n_iter     = %d\n", n_iter);
    printf("n_warmup   = %d\n", n_warmup);
  }

  fflush(stdout);

  test<gt::space::device, 0>(device_id, vendor_id, world_size, world_rank,
                             n_global, n_iter, true, 5);
  test<gt::space::device, 0>(device_id, vendor_id, world_size, world_rank,
                             n_global, n_iter, false, 5);
  test<gt::space::managed, 0>(device_id, vendor_id, world_size, world_rank,
                              n_global, n_iter, true, 5);
  test<gt::space::managed, 0>(device_id, vendor_id, world_size, world_rank,
                              n_global, n_iter, false, 5);

  test<gt::space::device, 1>(device_id, vendor_id, world_size, world_rank,
                             n_global, n_iter, true, 5);
  test<gt::space::device, 1>(device_id, vendor_id, world_size, world_rank,
                             n_global, n_iter, false, 5);
  test<gt::space::managed, 1>(device_id, vendor_id, world_size, world_rank,
                              n_global, n_iter, true, 5);
  test<gt::space::managed, 1>(device_id, vendor_id, world_size, world_rank,
                              n_global, n_iter, false, 5);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
