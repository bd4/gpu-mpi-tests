/*
 * Test GPU aware MPI on different platforms using a simple
 * distributed 1d stencil as an example. Gtensor is used so
 * a single source can be used for all platforms.
 */

#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "gtensor/gtensor.h"
#include "gtensor/reductions.h"

using namespace gt::placeholders;

// little hack to make code parameterizable on managed vs device memory
namespace gt {

namespace ext {
namespace detail {

template <typename T, gt::size_type N, typename S = gt::space::device>
struct gthelper {
  using gtensor = gt::gtensor<T, N, S>;
};

#ifdef GTENSOR_HAVE_DEVICE

template <typename T, gt::size_type N>
struct gthelper<T, N, gt::space::managed> {
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
 * Return unevaluated expression that calculates the stencil.
 *
 * Size of the result will be size of y minus 4 (the number of boundary points).
 */
inline auto stencil1d_5(const gt::gtensor_device<double, 1> &y,
                        const gt::gtensor<double, 1> &stencil) {
  return stencil(0) * y.view(_s(0, -4)) + stencil(1) * y.view(_s(1, -3)) +
         stencil(2) * y.view(_s(2, -2)) + stencil(3) * y.view(_s(3, -1)) +
         stencil(4) * y.view(_s(4, _));
}

void set_rank_device(int n_ranks, int rank) {
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

void boundary_exchange(MPI_Comm comm, int world_size, int rank,
                       gt::gtensor_device<double, 1> &d_y, int n_bnd) {
  double *d_y_data = gt::raw_pointer_cast(d_y.data());
  double *d_y_data_end = gt::raw_pointer_cast(d_y.data()) + d_y.size();

  MPI_Request req_l[2];
  MPI_Request req_r[2];

  int rank_l = rank - 1;
  int rank_r = rank + 1;

  if (rank_l >= 0) {
    printf("%d left\n", rank);
    // send/recv left boundary
    MPI_Irecv(d_y_data, n_bnd, MPI_DOUBLE, rank_l, 123, comm, &req_l[0]);
    MPI_Isend(d_y_data + n_bnd, n_bnd, MPI_DOUBLE, rank_l, 456, comm,
              &req_l[1]);
  }

  if (rank_r < world_size) {
    printf("%d right\n", rank);
    // send/recv right boundary
    MPI_Irecv(d_y_data_end - n_bnd, n_bnd, MPI_DOUBLE, rank_r, 456, comm,
              &req_r[0]);
    MPI_Isend(d_y_data_end - 2 * n_bnd, n_bnd, MPI_DOUBLE, rank_r, 123, comm,
              &req_r[1]);
  }

  int mpi_rval;
  if (rank_l >= 0) {
    printf("%d wait left\n", rank);
    mpi_rval = MPI_Waitall(2, req_l, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_l error: %d\n", mpi_rval);
    }
  }
  if (rank_r < world_size) {
    printf("%d wait right\n", rank);
    mpi_rval = MPI_Waitall(2, req_r, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("send_r error: %d\n", mpi_rval);
    }
  }
}

int main(int argc, char **argv) {
  constexpr int n_global = 8 * 1024 * 1024;
  constexpr int n_sten = 5;
  constexpr int n_bnd = (n_sten - 1) / 2;
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

  auto h_y = gt::empty<double>({n_local_with_ghost});
  auto d_y = gt::empty_device<double>({n_local_with_ghost});

  auto h_dydx_numeric = gt::empty<double>({n_local});
  auto h_dydx_actual = gt::empty<double>({n_local});
  auto d_dydx_numeric = gt::empty_device<double>({n_local});

  double lx = 8;
  double dx = lx / n_global;
  double lx_local = lx / world_size;
  double scale = n_global / lx;
  auto fn_x_cubed = [](double x) { return x * x * x; };
  auto fn_x_cubed_deriv = [](double x) { return 3 * x * x; };

  printf("%d Init\n", world_rank);
  double x_start = world_rank * lx_local;
  for (int i = 0; i < n_local; i++) {
    double xtmp = x_start + i * dx;
    h_y(i + n_bnd) = fn_x_cubed(xtmp);
    h_dydx_actual(i) = fn_x_cubed_deriv(xtmp);
  }

  // fill boundary points on ends
  if (world_rank == 1) {
    for (int i = 0; i < n_bnd; i++) {
      double xtmp = (i - n_bnd) * dx;
      h_y(i) = fn_x_cubed(xtmp);
    }
  }
  if (world_rank == world_size - 1) {
    for (int i = 0; i < n_bnd; i++) {
      double xtmp = lx + i * dx;
      h_y(n_bnd + n_local + i) = fn_x_cubed(xtmp);
    }
  }

  gt::copy(h_y, d_y);
  gt::synchronize();

  printf("%d Ex\n", world_rank);

  boundary_exchange(MPI_COMM_WORLD, world_size, world_rank, d_y, n_bnd);
  gt::synchronize();

  printf("%d Sten\n", world_rank);
  d_dydx_numeric = stencil1d_5(d_y, stencil5) * scale;
  gt::synchronize();

  printf("%d Copy\n", world_rank);
  gt::copy(d_dydx_numeric, h_dydx_numeric);
  gt::synchronize();

  /*
  for (int i = 0; i < 5; i++) {
    printf("{0} l {1}\n{0} l {2}\n", world_rank, h_dydx_actual(i),
               h_dydx_numeric(i));
  }
  for (int i = 0; i < 5; i++) {
    int idx = n_local - 1 - i;
    printf("{0} r {1}\n{0} r {2}\n", world_rank, h_dydx_actual(idx),
               h_dydx_numeric(idx));
  }
  */

  printf("%d Err calc\n", world_rank);
  double err_norm = std::sqrt(gt::sum_squares(h_dydx_numeric - h_dydx_actual));

  printf("%d/%d [%d:0x%08x] err_norm = %.8f\n", world_rank, world_size,
         device_id, vendor_id, err_norm);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
