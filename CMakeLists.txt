cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# create project
project(mpi-daxpy-test)

option(TEST_MANAGED "Test managed memory" ON)

# add dependencies
include(cmake/CPM.cmake)
CPMFindPackage(NAME gtensor
               GITHUB_REPOSITORY wdmapp/gtensor
               GIT_TAG "main"
               OPTIONS "GTENSOR_ENABLE_BLAS ON")

set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED)

if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
  enable_language(CUDA)
endif()

add_executable(mpi_daxpy_gt)
target_sources(mpi_daxpy_gt PRIVATE mpi_daxpy_gt.cc)
target_link_libraries(mpi_daxpy_gt PRIVATE gtensor::gtensor)
target_link_libraries(mpi_daxpy_gt PRIVATE gtensor::blas)
target_link_libraries(mpi_daxpy_gt PRIVATE MPI::MPI_CXX)

add_executable(mpi_stencil_gt)
target_sources(mpi_stencil_gt PRIVATE mpi_stencil_gt.cc)
target_link_libraries(mpi_stencil_gt PRIVATE gtensor::gtensor)
target_link_libraries(mpi_stencil_gt PRIVATE MPI::MPI_CXX)

add_executable(mpi_stencil2d_gt)
target_sources(mpi_stencil2d_gt PRIVATE mpi_stencil2d_gt.cc)
target_link_libraries(mpi_stencil2d_gt PRIVATE gtensor::gtensor)
target_link_libraries(mpi_stencil2d_gt PRIVATE MPI::MPI_CXX)
#target_compile_features(mpi_stencil2d_gt PRIVATE cxx_std_17)
if (TEST_MANAGED)
  message(STATUS "${PROJECT_NAME}: Enabling managed memory")
  target_compile_definitions(mpi_stencil2d_gt PRIVATE TEST_MANAGED)
endif()
if (GTENSOR_DEVICE STREQUAL "hip" AND DEFINED ENV{PE_MPICH_GTL_DIR_amd_gfx90a})
  message(STATUS "${PROJECT_NAME}: Linking gtl libs for HIP backend")
  target_link_options(mpi_stencil2d_gt PRIVATE
                      $ENV{PE_MPICH_GTL_DIR_amd_gfx90a}
                      $ENV{PE_MPICH_GTL_LIBS_amd_gfx90a})
endif()

if ("${GTENSOR_DEVICE}" STREQUAL "cuda") 
  set_source_files_properties(mpi_daxpy_gt.cc
                              TARGET_DIRECTORY mpi_daxpy_gt
                              PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(mpi_stencil_gt.cc
                              TARGET_DIRECTORY mpi_stencil_gt
                              PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(mpi_stencil2d_gt.cc
                              TARGET_DIRECTORY mpi_stencil2d_gt
                              PROPERTIES LANGUAGE CUDA)
else()
  set_source_files_properties(mpi_daxpy_gt.cc
                              TARGET_DIRECTORY mpi_daxpy_gt
                              PROPERTIES LANGUAGE CXX)
  set_source_files_properties(mpi_stencil_gt.cc
                              TARGET_DIRECTORY mpi_stencil_gt
                              PROPERTIES LANGUAGE CXX)
  set_source_files_properties(mpi_stencil2d_gt.cc
                              TARGET_DIRECTORY mpi_stencil2d_gt
                              PROPERTIES LANGUAGE CXX)
endif()

if ("${GTENSOR_DEVICE}" STREQUAL "sycl")
  add_executable(mpi_stencil2d_sycl)
  target_sources(mpi_stencil2d_sycl PRIVATE mpi_stencil2d_sycl.cc)
  target_link_libraries(mpi_stencil2d_sycl MPI::MPI_CXX)
  target_compile_options(mpi_stencil2d_sycl PRIVATE -fsycl -x c++)
  target_link_options(mpi_stencil2d_sycl PRIVATE -fsycl)

  add_executable(mpi_stencil2d_sycl_oo)
  target_sources(mpi_stencil2d_sycl_oo PRIVATE mpi_stencil2d_sycl_oo.cc)
  target_link_libraries(mpi_stencil2d_sycl_oo MPI::MPI_CXX)
  target_compile_options(mpi_stencil2d_sycl_oo PRIVATE -fsycl -x c++)
  target_link_options(mpi_stencil2d_sycl_oo PRIVATE -fsycl)
endif()
