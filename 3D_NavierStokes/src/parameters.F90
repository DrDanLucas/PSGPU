!     NOTES ON PARAMETERS: 
!
!     ISEED        ! Best to use numbers that are IKTX*IKTY apart 
!                  ! (for 256*256 this is ~15000)
!     v2           viscosity in dimensional terms, 1/Re in non-dimensional terms
!     KF           forcing wavenumber i.e forcing is sin(KF*y) in x-driection.
!     AMPV         initial enstrophy (initial condition is normalized according to AMPV)
!     outfreq1     time between outputs according to NOUT
!     outfreq2     time between outputs according to NOUTV
!                  
!      
!     UPOflag     ! set to 1 if TSTEP is to be read from Zk.in
!                  ! still set TSTEP above to be close to read in value
!                  ! so NOUT and NOUTV are suitable
!
      module PARAMETERS
!
      use GLOBAL_PARAMS 
      implicit none
!
      REAL(rk) :: Re,outfreq1,outfreq2
!
      contains
!
!------------------------------------------------
! Compute common groupings
!
      subroutine params_update()
      implicit none
!
      v2     =   1._rk / Re
      AMPFOR =  -0.25_rk*KF*KF
      NSTOP  =   INT(TSTEP/DELT) !Note this rounds down
      NOUT   =   INT(outfreq1/DELT)
      NOUTV  =   INT(outfreq2/DELT)
!
!     DEFINE CONSTANTS
!
      ZI = CMPLX(0._rk,1._rk,rk)
      TWOPI = 4._rk*ASIN(1._rk)
      ROOT2 = SQRT(2._rk)
!
      end subroutine params_update
!
!------------------------------------------------
!
      subroutine read_label(u, n)
      implicit none
!
      integer, intent(in) :: u
      character(len=*), intent(in) :: n
!
      integer :: i
      character :: t
!
      do i = 1,LEN(n)
         read(u,"(A1)",ADVANCE='NO') t
         if (t /= n(i:i)) then
            write(*,*) "bad label ", n
            stop
         endif
      enddo
!
      end subroutine read_label
!
!------------------------------------------------
!
      subroutine read_int(u, n, v)
      implicit none
!
      integer, intent(in) :: u
      character(len=*), intent(in) :: n
      integer, intent(out) :: v
!
      call read_label(u, n)
!
      read(u,"(I10)") v
      write(*,*) n, v
      end subroutine read_int
!
!------------------------------------------------
!
      subroutine read_logical(u, n, v)
      implicit none
!
      integer, intent(in) :: u
      character(len=*), intent(in) :: n
      logical, intent(out) :: v
!     
      call read_label(u, n)
!
      read(u,"(L1)") v
      write(*,*) n, v
      end subroutine read_logical
!
!------------------------------------------------
!
      subroutine read_float(u, n, v)
      implicit none
!
      integer, intent(in) :: u
      character(len=*), intent(in) :: n
      real(rk), intent(out) :: v
!      
      call read_label(u, n)
!      
      read(u,"(F12.4)") v
      write(*,*) n, v
      end subroutine read_float
!      
!------------------------------------------------
!
      subroutine read_string(u, n, v)
      implicit none
!      
      integer, intent(in) :: u
      character(len=*), intent(in) :: n
      character(len=*), intent(out) :: v
!      
      call read_label(u, n)
!      
      read(u,'(A256)') v
      write(*,*) n, v
      end subroutine read_string
!      
!------------------------------------------------
! Read parameters from a simple text file
!      
      subroutine params_read_txt()
      implicit none
!      
      open(55, FILE=trim(simdir)//'/params.txt', ACTION='READ')
!
      call read_int(55, 'ISEED = ', ISEED)
      call read_int(55, 'IREST = ', IREST)
      call read_string(55, 'infile = ', infile)
      call read_string(55, 'infile type = ', InfileType)
      call read_int(55, 'KF = ', KF)
      call read_float(55, 'Re = ', Re)
      call read_float(55, 'TSTEP = ', TSTEP)
      call read_float(55, 'ResidualThreshold = ', ResidualThreshold)
      call read_float(55, 'outfreq1 = ', outfreq1)
      call read_float(55, 'outfreq2 = ', outfreq2)
      call read_float(55, 'DELT = ', DELT)
      call read_float(55, 'AMPV = ', AMPV)              
      call read_int(55, 'STATSFLAG = ', STATSFLAG)
      call read_int(55, 'RCFLAG = ', RCFLAG)
      call read_int(55, 'UPOflag = ', UPOflag)
      call read_int(55, 'ISTOP = ', ISTOP)

!
      close(55)
!
      call params_update()
!
      end subroutine params_read_txt
!
!
!------------------------------------------------
!
      end module parameters
!
