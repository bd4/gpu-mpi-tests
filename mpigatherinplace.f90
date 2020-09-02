program mpigatherinplace
use mpi
implicit none

integer :: rank, ierr, nmpi, i

integer :: N, err
real(kind=8), dimension(:), allocatable :: allx
real :: asum, lsum

N = 128*1024*1024

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

allocate(allx(N*nmpi))

lsum = 0
do i=1, N
    allx(rank*N+i) = rank*i/N
    lsum = lsum + allx(rank*N+i)
end do

call MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, &
                 & allx, N, MPI_DOUBLE, MPI_COMM_WORLD, ierr)
if (ierr /= 0) then
    print *, 'Failed MPI_Allgather: ', ierr
    stop
end if

asum = sum(allx)

print *, rank, "/", nmpi, " ", lsum, " ", asum

deallocate(allx)

call MPI_Finalize(ierr)
if (ierr /= 0) then
    print *, 'Failed MPI_Finalize: ', ierr
    stop
end if

end program mpigatherinplace
