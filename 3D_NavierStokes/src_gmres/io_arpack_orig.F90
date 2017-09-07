module io_arpack
  use probsize
  implicit none

!
!     %----------------------%
!     | ARPACK Local Scalars |
!     %----------------------%
!
! dnaupd
      integer    bounds, ih, iq, ishift, iupd, iw,          & 
                 ldh, ldq, levec, mode, msglvl, mxiter, nb, &
                 nev0, next, np_2, ritzi, ritzr
      common /dnaupd_block/ bounds, ih, iq, ishift, iupd, iw, ldh, ldq,  &
                            levec, mode, msglvl, mxiter, nb, nev0, next, &
                            np_2, ritzi, ritzr

! dnaup2
      logical    cnorm , getv0, initv, update, ushift
      integer    iter_2 , kplusp, msglvl_2, nconv, &
                 nevbef, nev0_2 , np0  , numcnv
      Double precision  rnorm , eps23
      common /dnaup2_block/ rnorm , eps23, cnorm , getv0, initv,  &
                update, ushift, iter_2 , kplusp, msglvl_2, nconv, &
                nevbef, nev0_2 , np0  , numcnv
      save /dnaup2_block/

! dnapps
      logical    first
      Double precision ovfl, smlnum, ulp, unfl
      common /dnapps_block/ ovfl, smlnum, ulp, unfl, first
      save /dnapps_block/

! dgetv0
      logical    first_2, inits, orth
      integer    iseed(4), iter_3, msglvl_3
      Double precision   rnorm0
      common /dgetv0_block/ rnorm0, iseed, iter_3, msglvl_3, &
                  first_2, inits, orth 
      save /dgetv0_block/

! dnaitr
      logical    first_3, orth1, orth2, rstart, step3, step4
      integer    ierr_2, ipj, irj, ivj, iter_4, itry, j_2, msglvl_4
      Double precision betaj, ovfl_2, rnorm1, smlnum_2, ulp_2, unfl_2, wnorm
      common /dnaitr_block/  ovfl_2,                               &
                betaj, rnorm1, smlnum_2, ulp_2, unfl_2, wnorm,     &
                first_3, orth1, orth2, rstart, step3, step4,       &
                ierr_2, ipj, irj, ivj, iter_4, itry, j_2, msglvl_4  
      save /dnaitr_block/

contains

!------------------------------------------------
! Write a named dataset containing one (real)
! parameter into HDF5 group

  subroutine write_hdf5_float(name, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    real(rk), intent(in) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims = (/1/)
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dcreate_f(group_id, name, H5T_NATIVE_DOUBLE, dataspace_id, &
                      dataset_id, ierr)
    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, val, dims, ierr)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine write_hdf5_float

!------------------------------------------------
! Write a named dataset containing one (real)
! parameter into HDF5 group

  subroutine write_hdf5_float_array(name, lval, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: lval
    real(rk), dimension(lval), intent(in) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    dims(1) = lval
    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dcreate_f(group_id, name, H5T_NATIVE_DOUBLE, dataspace_id, &
                      dataset_id, ierr)
    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, val, dims, ierr)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine write_hdf5_float_array


!------------------------------------------------
! Write a named dataset containing one (real)
! parameter into HDF5 group

  subroutine write_hdf5_float_array2(name, lval1, lval2, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: lval1, lval2
    real(rk), dimension(lval1, lval2), intent(in) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(2) :: dims
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    dims(1) = lval1
    dims(2) = lval2
    call h5screate_simple_f(2, dims, dataspace_id, ierr)
    call h5dcreate_f(group_id, name, H5T_NATIVE_DOUBLE, dataspace_id, &
                      dataset_id, ierr)
    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, val, dims, ierr)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine write_hdf5_float_array2

!------------------------------------------------
! Write a dataset containing one (integer) parameter
! and a name into HDF5 group

  subroutine write_hdf5_int(name, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims = (/1/)
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dcreate_f(group_id, name, H5T_NATIVE_INTEGER, dataspace_id, &
                      dataset_id, ierr)
    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, val, dims, ierr)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine write_hdf5_int

!------------------------------------------------
! Write a dataset containing one (integer) parameter
! and a name into HDF5 group

  subroutine write_hdf5_int_array(name, lval, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: lval
    integer, dimension(lval), intent(in) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    dims(1) = lval
    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dcreate_f(group_id, name, H5T_NATIVE_INTEGER, dataspace_id, &
                      dataset_id, ierr)
    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, val, dims, ierr)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine write_hdf5_int_array
!------------------------------------------------
! Write a dataset containing one (integer) parameter
! and a name into HDF5 group

  subroutine write_hdf5_logical(name, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    logical, intent(in) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims = (/1/)
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: int, ierr

    if (val) then
       int = 1
    else
       int = 0
    end if
    
    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dcreate_f(group_id, name, H5T_NATIVE_INTEGER, dataspace_id, &
                      dataset_id, ierr)
    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, int, dims, ierr)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine write_hdf5_logical


!--------------------------------------------
! save ARPACK state

  subroutine arpack_state_write(ido, SzV, nev, TOL, resid, &
          ncv, V, ldv, iparam, ipntr, workd, workl, lworkl, info, cnt, switch)
    use hdf5
    use parameters
    use mesh
    implicit none

    integer, intent(in) :: SzV, nev, ncv, ldv, lworkl, ido, info
    integer, dimension(11), intent(in) :: iparam
    integer, dimension(14), intent(in) :: ipntr

    real(rk), intent(in) :: TOL
    real(rk), dimension(SzV), intent(in) :: resid
    real(rk), dimension(3*SzV), intent(in) :: workd
    real(rk), dimension(lworkl), intent(in) :: workl
    real(rk), dimension(SzV, ncv), intent(in) :: V

    integer, intent(in) :: cnt, switch

    integer(HID_T) :: file_id, group_id, group_id2
    character(len=6) :: numstr
    integer :: ierr

    write(numstr, '(I6.6)') switch

    call h5open_f(ierr)
    call h5fcreate_f('arpack'//numstr//'.h5', H5F_ACC_TRUNC_F, file_id, ierr)

    call params_write_hdf5(file_id)
    call mesh_write_hdf5(file_id)

    call write_hdf5_int('SzV', SzV, file_id)
    call write_hdf5_int('nev', nev, file_id)
    call write_hdf5_int('ncv', ncv, file_id)
    call write_hdf5_int('ldv', ldv, file_id)
    call write_hdf5_int('lworkl', lworkl, file_id)
    call write_hdf5_int('ido', ido, file_id)
    call write_hdf5_int('info', info, file_id)
    call write_hdf5_int('cnt', cnt, file_id)

    call write_hdf5_int_array('iparam', 11, iparam, file_id)
    call write_hdf5_int_array('ipntr', 14, ipntr, file_id)

    call write_hdf5_float('TOL', TOL, file_id)

    call write_hdf5_float_array('resid', SzV, resid, file_id)
    call write_hdf5_float_array('workd', 3*SzV, workd, file_id)
    call write_hdf5_float_array('workl', lworkl, workl, file_id)
    call write_hdf5_float_array2('V', SzV, ncv, V, file_id)


    call h5gcreate_f(file_id, "arpack_local_variables", group_id, ierr)

    call write_hdf5_int('bounds', bounds, group_id)
    call write_hdf5_int('ih', ih, group_id)
    call write_hdf5_int('iq', iq, group_id)
    call write_hdf5_int('ishift', ishift, group_id)
    call write_hdf5_int('iupd', iupd, group_id)
    call write_hdf5_int('iw', iw, group_id)
    call write_hdf5_int('ldh', ldh, group_id)
    call write_hdf5_int('ldq', ldq, group_id)
    call write_hdf5_int('levec', levec, group_id)
    call write_hdf5_int('mode', mode, group_id)
    call write_hdf5_int('msglvl', msglvl, group_id)
    call write_hdf5_int('mxiter', mxiter, group_id)
    call write_hdf5_int('nb', nb, group_id)
    call write_hdf5_int('nev0', nev0, group_id)
    call write_hdf5_int('next', next, group_id)
    call write_hdf5_int('np_2', np_2, group_id)
    call write_hdf5_int('ritzi', ritzi, group_id)
    call write_hdf5_int('ritzr', ritzr, group_id)
    call write_hdf5_int('iter_2', iter_2, group_id)
    call write_hdf5_int('kplusp', kplusp, group_id)
    call write_hdf5_int('msglvl_2', msglvl_2, group_id)
    call write_hdf5_int('nconv', nconv, group_id)
    call write_hdf5_int('nevbef', nevbef, group_id)
    call write_hdf5_int('nev0_2', nev0_2, group_id)  
    call write_hdf5_int('np0', np0, group_id)
    call write_hdf5_int('numcnv', numcnv, group_id)
    call write_hdf5_int('iter_3', iter_3, group_id)
    call write_hdf5_int('msglvl_3', msglvl_3, group_id)
    call write_hdf5_int('ierr_2', ierr_2, group_id)
    call write_hdf5_int('ipj', ipj, group_id)
    call write_hdf5_int('irj', irj, group_id)
    call write_hdf5_int('ivj', ivj, group_id)
    call write_hdf5_int('iter_4', iter_4, group_id)
    call write_hdf5_int('itry', itry, group_id)
    call write_hdf5_int('j_2', j_2, group_id)
    call write_hdf5_int('msglvl_4', msglvl_4, group_id)
    
    call write_hdf5_int_array('iseed', 4, iseed, file_id)

    call write_hdf5_float('rnorm', rnorm, group_id)
    call write_hdf5_float('eps23', eps23, group_id)
    call write_hdf5_float('smlnum', smlnum, group_id)
    call write_hdf5_float('ovfl', ovfl, group_id)
    call write_hdf5_float('ulp', ulp, group_id)
    call write_hdf5_float('unfl', unfl, group_id)
    call write_hdf5_float('rnorm0', rnorm0, group_id)
    call write_hdf5_float('betaj', betaj, group_id)
    call write_hdf5_float('ovfl_2', ovfl_2, group_id)
    call write_hdf5_float('rnorm1', rnorm1, group_id)
    call write_hdf5_float('smlnum_2', smlnum_2, group_id)
    call write_hdf5_float('ulp_2', ulp_2, group_id)
    call write_hdf5_float('unfl_2', unfl_2, group_id)
    call write_hdf5_float('wnorm', wnorm, group_id)

    call h5gcreate_f(file_id, "logical_variables_as_binary", group_id2, ierr)
    call write_hdf5_logical('cnorm', cnorm, group_id2)
    call write_hdf5_logical('getv0', getv0, group_id2)
    call write_hdf5_logical('initv', initv, group_id2)
    call write_hdf5_logical('update', update, group_id2)
    call write_hdf5_logical('ushift', ushift, group_id2)
    call write_hdf5_logical('first', first, group_id2)
    call write_hdf5_logical('first_2', first_2, group_id2)
    call write_hdf5_logical('inits', inits, group_id2)
    call write_hdf5_logical('orth', orth, group_id2)
    call write_hdf5_logical('first_3', first_3, group_id2)
    call write_hdf5_logical('orth1', orth1, group_id2)
    call write_hdf5_logical('orth2', orth2, group_id2)
    call write_hdf5_logical('rstart', rstart, group_id2)
    call write_hdf5_logical('step3', step3, group_id2)
    call write_hdf5_logical('step4', step4, group_id2)

    call h5fclose_f(file_id, ierr)
    call h5close_f(ierr)

  end subroutine arpack_state_write

!------------------------------------------------
! Read a dataset containing one (integer) parameter

  subroutine read_hdf5_float(name, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    real(rk), intent(out) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims = (/1/)
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dopen_f(group_id, name, dataset_id, ierr)
    call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, val, dims, ierr, &
         file_space_id = dataspace_id, mem_space_id = dataspace_id)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine read_hdf5_float


!------------------------------------------------
! Read a named dataset containing a 1D array of floats

  subroutine read_hdf5_float_array(name, lval, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: lval
    real(rk), dimension(lval), intent(out) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    dims(1) = lval
    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dopen_f(group_id, name, dataset_id, ierr)
    call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, val, dims, ierr, &
         file_space_id = dataspace_id, mem_space_id = dataspace_id)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine read_hdf5_float_array


!------------------------------------------------
! Read a named dataset containing a 2D array of floats

  subroutine read_hdf5_float_array2(name, lval1, lval2, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: lval1, lval2
    real(rk), dimension(lval1, lval2), intent(out) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(2) :: dims
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    dims(1) = lval1
    dims(2) = lval2
    call h5screate_simple_f(2, dims, dataspace_id, ierr)
    call h5dopen_f(group_id, name, dataset_id, ierr)
    call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, val, dims, ierr, &
         file_space_id = dataspace_id, mem_space_id = dataspace_id)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine read_hdf5_float_array2

!------------------------------------------------
! Read a dataset containing one (integer) parameter

  subroutine read_hdf5_int(name, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(out) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims = (/1/)
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dopen_f(group_id, name, dataset_id, ierr)
    call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, val, dims, ierr, &
         file_space_id = dataspace_id, mem_space_id = dataspace_id)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine read_hdf5_int

!------------------------------------------------
! Read a dataset containing a 1D array of integers

  subroutine read_hdf5_int_array(name, lval, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    integer, intent(in) :: lval
    integer, dimension(lval), intent(out) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: ierr

    dims(1) = lval
    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dopen_f(group_id, name, dataset_id, ierr)
    call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, val, dims, ierr, &
         file_space_id = dataspace_id, mem_space_id = dataspace_id)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

  end subroutine read_hdf5_int_array

!------------------------------------------------
! Read a dataset containing one (integer) parameter

  subroutine read_hdf5_logical(name, val, group_id)
    use hdf5
    implicit none

    character(len=*), intent(in) :: name
    logical, intent(out) :: val
    integer(HID_T), intent(in) :: group_id

    integer(HSIZE_T), dimension(1) :: dims = (/1/)
    integer(HID_T) :: dataset_id, dataspace_id
    integer :: int, ierr

    call h5screate_simple_f(1, dims, dataspace_id, ierr)
    call h5dopen_f(group_id, name, dataset_id, ierr)
    call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, int, dims, ierr, &
         file_space_id = dataspace_id, mem_space_id = dataspace_id)
    call h5dclose_f(dataset_id, ierr)
    call h5sclose_f(dataspace_id, ierr)

    if (int==1) then
       val = .true.
    else
       val = .false.
    end if

  end subroutine read_hdf5_logical
!--------------------------------------------
! read ARPACK state

  subroutine arpack_state_read(ido, SzV, nev, TOL, resid, &
          ncv, V, ldv, iparam, ipntr, workd, workl, lworkl, info, cnt, my_infile)
    use hdf5
    implicit none

    integer, intent(in) :: SzV, nev, ncv, ldv, lworkl
    integer, intent(out) :: ido, info, cnt
    integer, dimension(11), intent(out) :: iparam
    integer, dimension(14), intent(out) :: ipntr 
  
    real(rk), intent(in) :: TOL
    real(rk), dimension(SzV), intent(out) :: resid
    real(rk), dimension(3*SzV), intent(out) :: workd
    real(rk), dimension(lworkl), intent(out) :: workl
    real(rk), dimension(SzV, ncv), intent(out) :: V

    character(len=*), intent(in) :: my_infile
   
    integer(HID_T) :: file_id, group_id, group_id2
    character(len=6) :: numstr
    integer :: ierr

    call h5open_f(ierr)
    call h5fopen_f(my_infile, H5F_ACC_RDWR_F, file_id, ierr)

    call read_hdf5_int('ido', ido, file_id)
    call read_hdf5_int('info', info, file_id)
    call read_hdf5_int('cnt', cnt, file_id)

    call read_hdf5_int_array('iparam', 11, iparam, file_id)
    call read_hdf5_int_array('ipntr', 14, ipntr, file_id)

    call read_hdf5_float_array('resid', SzV, resid, file_id)
    call read_hdf5_float_array('workd', 3*SzV, workd, file_id)
    call read_hdf5_float_array('workl', lworkl, workl, file_id)
    call read_hdf5_float_array2('V', SzV, ncv, V, file_id)


    call h5gopen_f(file_id, "arpack_local_variables", group_id, ierr)

    call read_hdf5_int('bounds', bounds, group_id)
    call read_hdf5_int('ih', ih, group_id)
    call read_hdf5_int('iq', iq, group_id)
    call read_hdf5_int('ishift', ishift, group_id)
    call read_hdf5_int('iupd', iupd, group_id)
    call read_hdf5_int('iw', iw, group_id)
    call read_hdf5_int('ldh', ldh, group_id)
    call read_hdf5_int('ldq', ldq, group_id)
    call read_hdf5_int('levec', levec, group_id)
    call read_hdf5_int('mode', mode, group_id)
    call read_hdf5_int('msglvl', msglvl, group_id)
    call read_hdf5_int('mxiter', mxiter, group_id)
    call read_hdf5_int('nb', nb, group_id)
    call read_hdf5_int('nev0', nev0, group_id)
    call read_hdf5_int('next', next, group_id)
    call read_hdf5_int('np_2', np_2, group_id)
    call read_hdf5_int('ritzi', ritzi, group_id)
    call read_hdf5_int('ritzr', ritzr, group_id)
    call read_hdf5_int('iter_2', iter_2, group_id)
    call read_hdf5_int('kplusp', kplusp, group_id)
    call read_hdf5_int('msglvl_2', msglvl_2, group_id)
    call read_hdf5_int('nconv', nconv, group_id)
    call read_hdf5_int('nevbef', nevbef, group_id)
    call read_hdf5_int('nev0_2', nev0_2, group_id)  
    call read_hdf5_int('np0', np0, group_id)
    call read_hdf5_int('numcnv', numcnv, group_id)
    call read_hdf5_int('iter_3', iter_3, group_id)
    call read_hdf5_int('msglvl_3', msglvl_3, group_id)
    call read_hdf5_int('ierr_2', ierr_2, group_id)
    call read_hdf5_int('ipj', ipj, group_id)
    call read_hdf5_int('irj', irj, group_id)
    call read_hdf5_int('ivj', ivj, group_id)
    call read_hdf5_int('iter_4', iter_4, group_id)
    call read_hdf5_int('itry', itry, group_id)
    call read_hdf5_int('j_2', j_2, group_id)
    call read_hdf5_int('msglvl_4', msglvl_4, group_id)
    
    call read_hdf5_int_array('iseed', 4, iseed, file_id)

    call read_hdf5_float('rnorm', rnorm, group_id)
    call read_hdf5_float('eps23', eps23, group_id)
    call read_hdf5_float('smlnum', smlnum, group_id)
    call read_hdf5_float('ovfl', ovfl, group_id)
    call read_hdf5_float('ulp', ulp, group_id)
    call read_hdf5_float('unfl', unfl, group_id)
    call read_hdf5_float('rnorm0', rnorm0, group_id)
    call read_hdf5_float('betaj', betaj, group_id)
    call read_hdf5_float('ovfl_2', ovfl_2, group_id)
    call read_hdf5_float('rnorm1', rnorm1, group_id)
    call read_hdf5_float('smlnum_2', smlnum_2, group_id)
    call read_hdf5_float('ulp_2', ulp_2, group_id)
    call read_hdf5_float('unfl_2', unfl_2, group_id)
    call read_hdf5_float('wnorm', wnorm, group_id)

    call h5gopen_f(file_id, "logical_variables_as_binary", group_id2, ierr)
    call read_hdf5_logical('cnorm', cnorm, group_id2)
    call read_hdf5_logical('getv0', getv0, group_id2)
    call read_hdf5_logical('initv', initv, group_id2)
    call read_hdf5_logical('update', update, group_id2)
    call read_hdf5_logical('ushift', ushift, group_id2)
    call read_hdf5_logical('first', first, group_id2)
    call read_hdf5_logical('first_2', first_2, group_id2)
    call read_hdf5_logical('inits', inits, group_id2)
    call read_hdf5_logical('orth', orth, group_id2)
    call read_hdf5_logical('first_3', first_3, group_id2)
    call read_hdf5_logical('orth1', orth1, group_id2)
    call read_hdf5_logical('orth2', orth2, group_id2)
    call read_hdf5_logical('rstart', rstart, group_id2)
    call read_hdf5_logical('step3', step3, group_id2)
    call read_hdf5_logical('step4', step4, group_id2)

    call h5fclose_f(file_id, ierr)
    call h5close_f(ierr)

  end subroutine arpack_state_read


end module io_arpack
