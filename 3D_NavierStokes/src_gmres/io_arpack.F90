  module io_arpack
  use GLOBAL_PARAMS
  use PARAMETERS
  implicit none
!
  integer(kind=4), parameter :: SzV = 1842372  ! should be equal to 2*COUNT(LL)
!
!     %----------------------%
!     | ARPACK Local Scalars |
!     %----------------------%
!
!     These are common blocks I have added to ARPACK so that I can write out all of
!     ARPACK'S internally saved variables in binary. Reading all these variables back 
!     allows ARPACK to be restarted from precisely where it left off. 
!     See io_arpack_orig.F90 for my orginal routines using the hdf5 binary file format.      
!
!!$! dnaupd
!!$      integer    bounds, ih, iq, ishift, iupd, iw,          & 
!!$                 ldh, ldq, levec, mode, msglvl, mxiter, nb, &
!!$                 nev0, next, np_2, ritzi, ritzr
!!$      common /dnaupd_block/ bounds, ih, iq, ishift, iupd, iw, ldh, ldq,  &
!!$                            levec, mode, msglvl, mxiter, nb, nev0, next, &
!!$                            np_2, ritzi, ritzr
!!$
!!$! dnaup2
!!$      logical    cnorm , getv0, initv, update, ushift
!!$      integer    iter_2 , kplusp, msglvl_2, nconv, &
!!$                 nevbef, nev0_2 , np0  , numcnv
!!$      Double precision  rnorm , eps23
!!$      common /dnaup2_block/ rnorm , eps23, cnorm , getv0, initv,  &
!!$                update, ushift, iter_2 , kplusp, msglvl_2, nconv, &
!!$                nevbef, nev0_2 , np0  , numcnv
!!$      save /dnaup2_block/
!!$
!!$! dnapps
!!$      logical    first
!!$      Double precision ovfl, smlnum, ulp, unfl
!!$      common /dnapps_block/ ovfl, smlnum, ulp, unfl, first
!!$      save /dnapps_block/
!!$
!!$! dgetv0
!!$      logical    first_2, inits, orth
!!$      integer    iseed(4), iter_3, msglvl_3
!!$      Double precision   rnorm0
!!$      common /dgetv0_block/ rnorm0, iseed, iter_3, msglvl_3, &
!!$                  first_2, inits, orth 
!!$      save /dgetv0_block/
!!$
!!$! dnaitr
!!$      logical    first_3, orth1, orth2, rstart, step3, step4
!!$      integer    ierr_2, ipj, irj, ivj, iter_4, itry, j_2, msglvl_4
!!$      Double precision betaj, ovfl_2, rnorm1, smlnum_2, ulp_2, unfl_2, wnorm
!!$      common /dnaitr_block/  ovfl_2,                               &
!!$                betaj, rnorm1, smlnum_2, ulp_2, unfl_2, wnorm,     &
!!$                first_3, orth1, orth2, rstart, step3, step4,       &
!!$                ierr_2, ipj, irj, ivj, iter_4, itry, j_2, msglvl_4  
!!$      save /dnaitr_block/

 contains

!--------------------------------------------------------------------
  subroutine  workd2q(d,q)
    implicit none
!
    real(rk), dimension(SzV), intent(in) :: d
    COMPLEX(rk) :: q(3*NKT)
    integer :: ikt,it,cnt
!
    cnt = 1
    DO IT = 1,3
       DO IKT = 1, NKT
          IF(.NOT.LL(IKT))     CYCLE
          q((IT-1)*NKT+IKT) = CMPLX(d(cnt),d(cnt+1),rk)
          cnt = cnt + 2
       ENDDO
    ENDDO
!
    IF (cnt-1 .NE. SzV) THEN
       Print*,'ERROR: workd2q - final cnt is not equal to SzV'
       pRINT*,'cnt = ',cnt
       STOP
    END IF
!
  end subroutine  workd2q
!
!--------------------------------------------------------------------
  subroutine  v2workd(v,d)
    implicit none
!
    real(rk), dimension(SzV), intent(out) :: d
    COMPLEX(rk) :: v(3*NKT)
    integer :: ikt,it,cnt
!
    cnt = 1
    DO IT = 1,3
       DO IKT = 1, NKT
          IF(.NOT.LL(IKT))     CYCLE
          d(cnt) = REAL(v((IT-1)*NKT+IKT))
          d(cnt+1) = AIMAG(v((IT-1)*NKT+IKT))
          cnt = cnt + 2
       ENDDO
    ENDDO
!
    IF (cnt-1 .NE. SzV) THEN
       Print*,'ERROR: v2workd - final cnt is not equal to SzV'
       pRINT*,'cnt = ',cnt
       STOP
    END IF
!
  end subroutine  v2workd
!
!--------------------------------------------------------------------
  subroutine arpack_final_output(info,nev,ncv,which,iparam,sel,      &
     &  DR, DI,Varn, SzV, workev, TOL,resid, ipntr, workd, workl,lworkl,T )
      
    integer :: SzV,nev,ncv,lworkl
!
      real(rk) :: TOL 
!
      integer(kind=4) :: info, i
      character(len=2) :: which
!
      integer(kind=4), dimension(11) :: iparam
      integer(kind=4), dimension(14) :: ipntr
      logical, dimension(ncv) :: sel
!
      real(rk), dimension(SzV) :: resid
      real(rk), dimension(SzV, ncv) :: Varn
      real(rk), dimension(3*SzV) :: workd
      real(rk), dimension(lworkl) :: workl
!
      real(rk), dimension(3*ncv) :: workev
      real(rk), dimension(nev+1) :: DR, DI
!
      real(rk) :: T
!
    if (info < 0) then

       print *, ' '
       print *, ' Error with _naupd, info = ',info
       print *, ' Check the documentation of _naupd'
       print *, ' '

    else

       write(*,*) 'nev = ', nev
       write(*,*) 'ncv = ', ncv
       write(*,*) 'which = ', which
       write(*,*) ' '
       write(*,*) 'If info in/out = 0: Normal exit.'
       write(*,*) 'info in = ', info
       write(*,*) 'IPARAM(5) = NCONV: number of "converged" Ritz values.'
       write(*,*) 'iparam(5) = ', iparam(5)
       if (iparam(5) .EQ. nev) then
          call dneupd(.true., 'A', sel, DR, DI, Varn, SzV, &
            0._rk, 0._rk, workev, 'I', SzV, which, nev, TOL, &
            resid, ncv, Varn, SzV, iparam, ipntr, workd, workl, &
            lworkl, info)
          write(*,*) 'info out = ', info
          write(*,*) 'iparam(5) = ', iparam(5)
          write(*,*) ' '
          write(*,*) 'Ritz estimates: ( Re , Im )'
          do i = 1,iparam(5)   
             write(*,*), i,': (', DR(i), ',', DI(i), ')'
          enddo

          write(*,*) ' '
          write(*,*) 'Ritz estimates: ( ln(| |)/T , arg()/T )'
          do i = 1,iparam(5)
             write(*,*), i,': (', LOG(SQRT(DR(i)*DR(i)+DI(i)*DI(i)))/T,',', ATAN2(DI(i),DR(i))/T,')'   
             !     call v2q(V(1,i), q)
             !     call simstate_write(q, trim(outfile1), i)
             !      call dsigdqbar(q, qbar)
             !      call simstate_write(q, trim(outfile2), i)
          enddo

          write(*,*) 'arg()/(2*PI) = fraction of eigenmode period swept through by A*v'
          write(*,*) ' '
          write(*,*) 'The complex Ritz vector associated with the Ritz value '
          write(*,*) 'with positive imaginary part is stored in two consecutive '
          write(*,*) 'columns of Varn.  The first column holds the real part of the Ritz '
          write(*,*) 'vector and the second column holds the imaginary part.  The '
          write(*,*) 'Ritz vector associated with the Ritz value with negative '
          write(*,*) 'imaginary part is simply the complex conjugate of the Ritz vector ' 
          write(*,*) 'associated with the positive imaginary part. '
       else
          write(*,*) 'number of converged ritz values is different to nev'
          write(*,*) 'aborting call to dneupd'
       endif
    endif
    
  end subroutine arpack_final_output

!------------------------------------------------------------------------



end module io_arpack
