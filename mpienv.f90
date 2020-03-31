program mpienv
use mpi
use iso_c_binding
implicit none

integer :: rank, ierr, nmpi

integer :: strlen, err, memory_per_core
character(len=5) :: read_env

call MPI_Init(ierr)
if (ierr /= 0) then
    print *, 'Failed MPI_Init: ', ierr
    stop
end if

call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
if (ierr /= 0) then
    print *, 'Failed MPI_COMM_RANK: ', ierr
    stop
end if

call MPI_COMM_SIZE(MPI_COMM_WORLD, nmpi, ierr)
if (ierr /= 0) then
    print *, 'Failed MPI_COMM_SIZE: ', ierr
    stop
end if

call get_environment_variable('MEMORY_PER_CORE',read_env,strlen,err)
read(read_env, '(i6)') memory_per_core

print *, 'rank ', rank, ' MEMORY_PER_CORE=', memory_per_core


end program mpienv
