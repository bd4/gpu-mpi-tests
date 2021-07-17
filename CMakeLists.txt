cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# create project
project(mpi-daxpy-test)

set(GTENSOR_ENABLE_BLAS ON CACHE BOOL "Enable gtblas")

# add dependencies
include(cmake/CPM.cmake)
CPMFindPackage(NAME gtensor
               GITHUB_REPOSITORY wdmapp/gtensor
               GIT_TAG main
               OPTIONS "GTENSOR_ENABLE_BLAS ON")

find_package(MPI REQUIRED)

add_executable(mpi_daxpy_gt)
target_sources(mpi_daxpy_gt PRIVATE mpi_daxpy_gt.cc)
target_link_libraries(mpi_daxpy_gt gtensor::gtensor)
target_link_libraries(mpi_daxpy_gt gtensor::blas)
target_link_libraries(mpi_daxpy_gt MPI::MPI_CXX)

if ("${GTENSOR_DEVICE}" STREQUAL "cuda") 
  enable_language(CUDA)
  set_source_files_properties(mpi_daxpy_gt.cc
                              TARGET_DIRECTORY mpi_daxpy_gt
                              PROPERTIES LANGUAGE CUDA)
else()
  set_source_files_properties(mpi_daxpy_gt.cc
                              TARGET_DIRECTORY mpi_daxpy_gt
                              PROPERTIES LANGUAGE CXX)
endif()
