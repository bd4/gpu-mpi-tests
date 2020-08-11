#!/bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: $0 um|noum nsys|nvprof|none nodes ppn"
  exit 1
fi

um=$1
prof=$2
nodes=$3
ppn=$4

tag=${um}_${prof}_${nodes}_${ppn}

if [ $prof == "nsys" ]; then
  prof_cmd="nsys profile --kill=none -c cudaProfilerApi -o profile/${tag}.%q{PMIX_RANK}"
elif [ $prof == "nvprof" ]; then
  prof_cmd="nvprof -o profile/nvprof.%q{PMIX_RANK}.nvvp --profile-from-start off"
else
  prof_cmd=""
fi

if [ $um == "um" ]; then
  cmd=./mpi_daxpy_nvtx_managed
else
  cmd=./mpi_daxpy_nvtx_unmanaged
fi

total_procs=$((ppn * nodes))

set +x
mpirun -np $total_procs \
   $prof_cmd $cmd >out-${tag}.txt 2>&1
set -x
