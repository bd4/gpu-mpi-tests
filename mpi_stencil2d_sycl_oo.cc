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
 * Modified version that uses minimal owning (array2d) and non-owning (span2d)
 * classes to make indexing handling less error prone, without using all of
 * gtensor. Note that the owning class is not trivially copyable and not device
 * copyable, because it must have a non-trivial destructor.
 *
 * TODO: Since no temporaries are used, perhaps a helper that allocates and
 * returns a span is a simpler option to create this minimal example?
 */

#include <cassert>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <type_traits>

#include "sycl/sycl.hpp"

// #define DEBUG

#ifdef DEBUG
#define dprintf(...) fprintf(stderr, __VA_ARGS__)
#else
#define dprintf(...)                                                           \
  do {                                                                         \
  } while (0)
#endif

constexpr std::size_t idx2(int n, int row, int col)
{
  return row + col * n;
}

template <typename T, sycl::usm::alloc Alloc>
class span2d
{
public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;

  span2d(T* data, const int nrows, const int ncols)
    : data_(data), nrows_(nrows), ncols_(ncols)
  {}

  // use default copy and move ctor. Ideall move ctor would better invalidate
  // the moved from object, but this is supposed to be a small example...
  span2d(const span2d& other) = default;
  span2d& operator=(const span2d& other) = default;
  span2d(span2d&& other) = default;
  span2d& operator=(span2d&& other) = default;

  // Note: shallow const
  reference operator()(int row, int col) const
  {
    assert(row < nrows_);
    assert(col < ncols_);
    return data_[idx2(nrows_, row, col)];
  }

  // Note: shallow const
  reference operator[](size_type i) const
  {
    assert(i < (nrows_ * ncols_));
    return data_[i];
  }

  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  size_type size() const { return nrows_ * ncols_; }

  span2d to_span() { return *this; }

  // Note: shallow const
  pointer data() const { return data_; }

private:
  const sycl::usm::alloc alloc_ = Alloc;
  T* data_;
  const int nrows_;
  const int ncols_;
};

template <typename T>
auto empty_host(sycl::queue& q, int nrows, int ncols)
{
  T* data = sycl::malloc(nrows * ncols, q, sycl::usm::alloc::host);
  return span2d<T, sycl::usm::alloc::host>(data, nrows, ncols);
}

template <typename T>
auto empty_device(sycl::queue& q, int nrows, int ncols)
{
  T* data = sycl::malloc(nrows * ncols, q, sycl::usm::alloc::device);
  return span2d<T, sycl::usm::alloc::device>(data, nrows, ncols);
}

template <typename T, sycl::usm::alloc Alloc>
class array2d : public span2d<T, Alloc>
{
public:
  using base_type = span2d<T, Alloc>;
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;

  array2d(sycl::queue& q, const int nrows, const int ncols)
    : base_type(sycl::malloc<value_type>(nrows * ncols, q, Alloc), nrows,
                ncols),
      q_(q)
  {}

  // Results in a double free, why?
  // ~array2d() { sycl::free(this->data(), q_); }

  // skip these to keep the example simple, pass by reference everywhere
  array2d(const array2d& other) = delete;
  array2d& operator=(const array2d& other) = delete;
  array2d(array2d&& other) = delete;
  array2d& operator=(array2d&& other) = delete;

  base_type to_span()
  {
    return base_type(this->data(), this->nrows(), this->ncols());
  }

private:
  sycl::queue& q_;
};

template <typename SrcArray, typename DestArray>
auto copy(sycl::queue& q, SrcArray& src, DestArray& dest)
{
  static_assert(std::is_same_v<typename SrcArray::value_type,
                               typename DestArray::value_type>,
                "value types must match");
  assert(src.size() == dest.size());
  return q.copy(src.data(), dest.data(), dest.size());
}

template <typename Array>
auto copy_dest_slice(sycl::queue& q, Array& src, Array& dest, int dim,
                     int start, int end)
{
  dprintf("copy dest_slice %d %d %d\n", dim, start, end);
  auto s_src = src.to_span();
  auto s_dest = dest.to_span();
  assert(dim == 0 || dim == 1);
  if (dim == 0) {
    assert(src.ncols() == dest.ncols());
    if (start < 0) {
      start += dest.nrows();
    }
    if (end < 0) {
      end += dest.nrows();
    } else if (end == 0 && start > end) {
      end = dest.nrows();
    }
    assert(start < end);
    auto range = sycl::range<2>(dest.ncols(), end - start);
    dprintf("d_z < buf range %d - %d (%d, %d)\n", start, end, dest.ncols(),
            end - start);
    auto e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(range, [=](sycl::item<2> item) {
        int row = item.get_id(1);
        int col = item.get_id(0);
        s_dest(start + row, col) = s_src(row, col);
      });
    });
    return e;
  } else {
    assert(src.nrows() == dest.nrows());
    if (start < 0) {
      start += dest.ncols();
    }
    if (end < 0) {
      end += dest.ncols();
    } else if (end == 0 && start > end) {
      end = dest.ncols();
    }
    auto range = sycl::range<2>(end - start, dest.nrows());
    auto e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(range, [=](sycl::item<2> item) {
        int row = item.get_id(1);
        int col = item.get_id(0);
        s_dest(row, start + col) = s_src(row, col);
      });
    });
    return e;
  }
}

template <typename Array>
auto copy_src_slice(sycl::queue& q, Array& src, Array& dest, int dim, int start,
                    int end)
{
  dprintf("copy  src_slice %d %d %d (%d, %d) -> (%d, %d)\n", dim, start, end,
          src.nrows(), src.ncols(), dest.nrows(), dest.ncols());
  assert(dim == 0 || dim == 1);
  auto s_src = src.to_span();
  auto s_dest = dest.to_span();
  if (dim == 0) {
    assert(src.ncols() == dest.ncols());
    if (start < 0) {
      start += src.nrows();
    }
    if (end < 0) {
      end += src.nrows();
    } else if (end == 0 && start > end) {
      end = src.nrows();
    }
    auto range = sycl::range<2>(dest.ncols(), end - start);
    dprintf("buf < d_z range %d - %d (%d, %d)\n", start, end, dest.ncols(),
            end - start);
    auto e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(range, [=](sycl::item<2> item) {
        int row = item.get_id(1);
        int col = item.get_id(0);
        s_dest(row, col) = s_src(start + row, col);
      });
    });
    return e;
  } else {
    assert(src.nrows() == dest.nrows());
    if (start < 0) {
      start += src.ncols();
    }
    if (end < 0) {
      end += src.ncols();
    } else if (end == 0 && start > end) {
      end = src.ncols();
    }
    auto range = sycl::range<2>(end - start, dest.nrows());
    auto e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(range, [=](sycl::item<2> item) {
        int row = item.get_id(1);
        int col = item.get_id(0);
        s_dest(row, col) = s_src(row, start + col);
      });
    });
    return e;
  }
}

inline void check(const char* fname, int line, int mpi_rval)
{
  if (mpi_rval != MPI_SUCCESS) {
    printf("%s:%d error %d\n", fname, line, mpi_rval);
    exit(2);
  }
}

#define CHECK(x) check(__FILE__, __LINE__, (x))

static constexpr double stencil5[] = {1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0,
                                      -1.0 / 12.0};

/*
 * Calculate 1d stencil of second dimension of 2d array on GPU. Out array must
 * be contiguous column major nrows x ncols array, while in array must be
 * (nrows)x(ncols+4) to accomodate 2 ghost points in each direction for the
 * second dimension.
 *
 * Returns sycl event, async with respect to host.
 */
template <typename Array>
auto stencil2d_1d_5(sycl::queue& q, Array& out2d, Array& in2d, double scale)
{
  // Note: swap index order; SYCL is row-major oriented, and this example
  // is col-major
  auto range = sycl::range<2>(out2d.ncols(), out2d.nrows());
  auto s_in2d = in2d.to_span();
  auto s_out2d = out2d.to_span();
  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range, [=](sycl::item<2> item) {
      int row = item.get_id(1);
      int col = item.get_id(0);
      s_out2d(row, col) = (stencil5[0] * s_in2d(row + 0, col) +
                           stencil5[1] * s_in2d(row + 1, col) +
                           stencil5[2] * s_in2d(row + 2, col) +
                           stencil5[3] * s_in2d(row + 3, col) +
                           stencil5[4] * s_in2d(row + 4, col)) *
                          scale;
    });
  });
  return e;
}

/*
 * Calculate the norm of the difference of two arrays, as sqrt of sum of
 * squared distances.
 */
double diff_norm(sycl::queue& q, std::size_t size, double* d_a, double* d_b)
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

  sycl::context ctx{};
  auto devices = ctx.get_devices();
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

  dprintf("%d: n_devices = %d\n", rank, n_devices);
  dprintf("%d: device_idx = %d\n", rank, device_idx);

  return sycl::queue{devices[device_idx], sycl::property::queue::in_order()};
}

// exchange in first dimension, staging into contiguous buffers on device
template <typename T, sycl::usm::alloc Alloc>
void boundary_exchange_x(MPI_Comm comm, int world_size, int rank,
                         sycl::queue& q, int n_bnd, array2d<T, Alloc>& d_z,
                         bool stage_host = false)
{
  static array2d<double, sycl::usm::alloc::device> sbuf_l{q, n_bnd,
                                                          d_z.ncols()};
  static array2d<double, sycl::usm::alloc::device> sbuf_r{q, n_bnd,
                                                          d_z.ncols()};
  static array2d<double, sycl::usm::alloc::device> rbuf_l{q, n_bnd,
                                                          d_z.ncols()};
  static array2d<double, sycl::usm::alloc::device> rbuf_r{q, n_bnd,
                                                          d_z.ncols()};

  static array2d<double, sycl::usm::alloc::host> h_sbuf_l{q, n_bnd,
                                                          d_z.ncols()};
  static array2d<double, sycl::usm::alloc::host> h_sbuf_r{q, n_bnd,
                                                          d_z.ncols()};
  static array2d<double, sycl::usm::alloc::host> h_rbuf_l{q, n_bnd,
                                                          d_z.ncols()};
  static array2d<double, sycl::usm::alloc::host> h_rbuf_r{q, n_bnd,
                                                          d_z.ncols()};

  int buf_size = sbuf_l.size();

  MPI_Request req_l[2];
  MPI_Request req_r[2];

  int rank_l = rank - 1;
  int rank_r = rank + 1;

  // start async copy of ghost points into send buffers
  if (rank_l >= 0) {
    dprintf("%d: rank_l = %d\n", rank, rank_l);
    fflush(nullptr);
    // sbuf_l = d_z.view(_all, _s(n_bnd, 2 * n_bnd));
    auto e = copy_src_slice(q, d_z, sbuf_l, 0, n_bnd, 2 * n_bnd);
    if (stage_host) {
      e.wait();
      copy(q, sbuf_l, h_sbuf_l).wait();
      for (int i = 0; i < h_sbuf_l.ncols(); i++) {
        for (int j = 0; j < h_sbuf_l.nrows(); j++) {
          dprintf("%d: sbuf_l[%d, %d] = %f\n", rank, j, i, h_sbuf_l(j, i));
          fflush(nullptr);
        }
      }
    }
  }
  if (rank_r < world_size) {
    dprintf("%d: rank_r = %d\n", rank, rank_r);
    fflush(nullptr);
    // sbuf_r = d_z.view(_all, _s(-2 * n_bnd, -n_bnd));
    auto e = copy_src_slice(q, d_z, sbuf_r, 0, -2 * n_bnd, -n_bnd);
    if (stage_host) {
      e.wait();
      copy(q, sbuf_r, h_sbuf_r).wait();
      for (int i = 0; i < h_sbuf_r.ncols(); i++) {
        for (int j = 0; j < h_sbuf_r.nrows(); j++) {
          dprintf("%d: sbuf_r[%d, %d] = %f\n", rank, j, i, h_sbuf_r(j, i));
          fflush(nullptr);
        }
      }
    }
  }

  // initiate async recv
  if (rank_l >= 0) {
    double* rbuf_l_data = nullptr;
    if (stage_host) {
      rbuf_l_data = h_rbuf_l.data();
    } else {
      rbuf_l_data = rbuf_l.data();
    }
    MPI_Irecv(rbuf_l_data, buf_size, MPI_DOUBLE, rank_l, 123, comm, &req_l[0]);
  }

  if (rank_r < world_size) {
    double* rbuf_r_data = nullptr;
    if (stage_host) {
      rbuf_r_data = h_rbuf_r.data();
    } else {
      rbuf_r_data = rbuf_r.data();
    }
    MPI_Irecv(rbuf_r_data, buf_size, MPI_DOUBLE, rank_r, 456, comm, &req_r[0]);
  }

  // wait for send buffer fill
  q.wait();

  // initiate async sends
  if (rank_l >= 0) {
    double* sbuf_l_data = nullptr;
    if (stage_host) {
      sbuf_l_data = h_sbuf_l.data();
    } else {
      sbuf_l_data = sbuf_l.data();
    }
    MPI_Isend(sbuf_l_data, buf_size, MPI_DOUBLE, rank_l, 456, comm, &req_l[1]);
  }

  if (rank_r < world_size) {
    double* sbuf_r_data = nullptr;
    if (stage_host) {
      sbuf_r_data = h_sbuf_r.data();
    } else {
      sbuf_r_data = sbuf_r.data();
    }
    MPI_Isend(sbuf_r_data, buf_size, MPI_DOUBLE, rank_r, 123, comm, &req_r[1]);
  }

  // wait for send/recv to complete, then copy data back into main data array
  int mpi_rval;
  if (rank_l >= 0) {
    mpi_rval = MPI_Waitall(2, req_l, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("%d: send_l error: %d\n", rank, mpi_rval);
    }
    if (stage_host) {
#ifdef DEBUG
      for (int i = 0; i < h_rbuf_l.ncols(); i++) {
        for (int j = 0; j < h_rbuf_l.nrows(); j++) {
          dprintf("%d: rbuf_l[%d, %d] = %f\n", rank, j, i, h_rbuf_l(j, i));
          fflush(nullptr);
        }
      }
#endif
      copy(q, h_rbuf_l, rbuf_l).wait();
    }
    // d_z.view(_all, _s(0, n_bnd)) = rbuf_l;
    copy_dest_slice(q, rbuf_l, d_z, 0, 0, n_bnd);
  }
  if (rank_r < world_size) {
    mpi_rval = MPI_Waitall(2, req_r, MPI_STATUSES_IGNORE);
    if (mpi_rval != MPI_SUCCESS) {
      printf("%d: send_r error: %d\n", rank, mpi_rval);
    }
    if (stage_host) {
#ifdef DEBUG
      for (int i = 0; i < h_rbuf_r.ncols(); i++) {
        for (int j = 0; j < h_rbuf_r.nrows(); j++) {
          dprintf("%d: rbuf_r[%d, %d] = %f\n", rank, j, i, h_rbuf_r(j, i));
          fflush(nullptr);
        }
      }
#endif
      copy(q, h_rbuf_r, rbuf_r).wait();
    }
    // d_z.view(_all, _s(-n_bnd, _)) = rbuf_r;
    copy_dest_slice(q, rbuf_r, d_z, 0, -n_bnd, 0);
  }

  q.wait();
}

int main(int argc, char** argv)
{
  using T = double;

  static_assert(
    std::is_trivially_copyable_v<span2d<T, sycl::usm::alloc::device>>,
    "span2d device not trivial");
  static_assert(std::is_trivially_copyable_v<span2d<T, sycl::usm::alloc::host>>,
                "span2d host not trivial");

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

#ifdef DEBUG
  n_global /= 1024;
  n_iter = 1;
  n_warmup = 0;
#endif

  int n_sten = 5;
  int n_bnd = (n_sten - 1) / 2;
  int world_size, world_rank, device_id;
  uint32_t vendor_id;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (n_global % world_size != 0) {
    printf("%d: nmpi (%d) must be divisor of domain size (%d), exiting\n",
           world_rank, world_size, n_global);
    exit(1);
  }

  const int n_local = n_global / world_size;
  const int n_local_with_ghost = n_local + 2 * n_bnd;

  sycl::queue q = get_rank_queue(world_size, world_rank);

  vendor_id = q.get_device().get_info<sycl::info::device::vendor_id>();

  if (world_rank == 0) {
    printf("n procs    = %d\n", world_size);
    printf("rank       = %d\n", world_rank);
    printf("n_global   = %d\n", n_global);
    printf("n_local    = %d\n", n_local);
    printf("n_iter     = %d\n", n_iter);
    printf("n_warmup   = %d\n", n_warmup);
    printf("stage_host = %d\n", stage_host);
  }

  int z_size = n_local_with_ghost * n_global;
  int dzdx_size = n_local * n_global;

  array2d<T, sycl::usm::alloc::host> h_z{q, n_local_with_ghost, n_global};
  array2d<T, sycl::usm::alloc::device> d_z{q, n_local_with_ghost, n_global};

  array2d<T, sycl::usm::alloc::host> h_dzdx_actual{q, n_local, n_global};
  array2d<T, sycl::usm::alloc::host> h_dzdx_numeric{q, n_local, n_global};
  array2d<T, sycl::usm::alloc::device> d_dzdx_actual{q, n_local, n_global};
  array2d<T, sycl::usm::alloc::device> d_dzdx_numeric{q, n_local, n_global};

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
  for (int j = 0; j < h_z.ncols(); j++) {
    double ytmp = j * dx;
    for (int i = 0; i < n_local; i++) {
      double xtmp = x_start + i * dx;
      h_z(i + n_bnd, j) = fn(xtmp, ytmp);
      h_dzdx_actual(i, j) = fn_dzdx(xtmp, ytmp);
    }
  }

  // fill boundary points on ends
  if (world_rank == 0) {
    for (int j = 0; j < h_z.ncols(); j++) {
      double ytmp = j * dx;
      for (int i = 0; i < n_bnd; i++) {
        double xtmp = (i - n_bnd) * dx;
        h_z(i, j) = fn(xtmp, ytmp);
      }
    }
  }
  if (world_rank == world_size - 1) {
    for (int j = 0; j < h_z.ncols(); j++) {
      double ytmp = j * dx;
      for (int i = 0; i < n_bnd; i++) {
        double xtmp = lx + i * dx;
        h_z(n_bnd + n_local + i, j) = fn(xtmp, ytmp);
      }
    }
  }

#ifdef DEBUG
  for (int r = 0; r < world_size; r++) {
    if (r != world_rank) {
      continue;
    }

    for (int i = n_bnd; i < 2 * n_bnd; i++) {
      dprintf("%d: [%d, :]", world_rank, i);
      for (int j = 0; j < std::min(20, h_z.ncols()); j++) {
        dprintf(" %f", h_z(i, j));
      }
      dprintf("\n");
    }
    for (int i = h_z.nrows() - 2 * n_bnd; i < h_z.nrows() - n_bnd; i++) {
      dprintf("%d: [%d, :]", world_rank, i);
      for (int j = 0; j < std::min(20, h_z.ncols()); j++) {
        dprintf(" %f", h_z(i, j));
      }
      dprintf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  copy(q, h_z, d_z);

  for (int i = 0; i < n_warmup + n_iter; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    boundary_exchange_x(MPI_COMM_WORLD, world_size, world_rank, q, n_bnd, d_z,
                        stage_host);
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9);

    if (i >= n_warmup) {
      total_time += iter_time;
    }

    // do some calculation, to try to more closely simulate what happens in GENE
    auto e = stencil2d_1d_5(q, d_dzdx_numeric, d_z, scale);
    e.wait();
  }
  printf("%d: exchange time %0.8f ms\n", world_rank,
         total_time / n_iter * 1000);

  copy(q, d_dzdx_numeric, h_dzdx_numeric).wait();

  /*
  for (int i = 0; i < 5; i++) {
    int idx = idx2(n_global, 8, i);
    printf("%d la %f\n%d ln %f\n", world_rank, h_dzdx_actual[idx], world_rank,
           h_dzdx_numeric[idx]);
  }
  for (int i = 0; i < 5; i++) {
    int idx = idx2(n_global, 8, n_local - 1 - i);
    printf("%d ra %f\n%d rn %f\n", world_rank, h_dzdx_actual[idx], world_rank,
           h_dzdx_numeric[idx]);
  }
  */

  double err_norm = diff_norm(q, h_dzdx_numeric.size(), h_dzdx_numeric.data(),
                              h_dzdx_actual.data());

  printf("%d: [0x%08x] err_norm = %.8f\n", world_rank, vendor_id, err_norm);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
