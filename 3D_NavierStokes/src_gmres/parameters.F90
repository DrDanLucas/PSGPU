!
!
!     NOTES ON PARAMETERS: 
!
!     ISEED        ! Best to use numbers that are IKTX*IKTY apart 
!                  ! (for 256*256 this is ~15000)
!     v2           viscosity in dimensional terms, 1/Re in non-dimensional terms
!     KF           forcing wavenumber i.e forcing is sin(KF*y) in x-driection.
!     AMPV         initial enstrophy (initial condition is normalized according to AMPV)
!     outfreq1     time between outputs according to NOUT
!     outfreq2     time between outputs according to NOUTV
!     NSHIFT       number of different evenly spacecd shifts tried 
!                  when finding UPO guesses.
!                  
!      
!     ICFLAG       ! stops code after generating i.c.'s
!     WNFLAG       ! If irest.NE.0 then WNFLAG determines whether or not 
!                  ! to add white noise to the starting state
!     SubSpaceFLAG ! Set to 1 to work in symmetric subspace
!     lamFLAG      ! sets up initial condition and gives results as 
!                  ! deviations away form the laminar solution
!     UPOflag      ! set to 1 if TSTEP is to be read from Zk.in
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
      AMPV   =   1._rk
      NSTOP  =   INT(TSTEP/DELT) !Note this rounds down
      NOUT   =   INT(outfreq1/DELT)
      NOUTV  =   INT(outfreq2/DELT)
      AMPFOR =   -0.25_rk*REAL(KF,rk)
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
      open(55, FILE='params.txt', ACTION='READ')
!
      call read_int(55, 'ISEED = ', ISEED)
      call read_int(55, 'IREST = ', IREST)
      call read_string(55, 'infile = ', infile)                   
      call read_string(55, 'infile type = ', InfileType)  
      call read_float(55, 'Re = ', Re)
      call read_float(55, 'TSTEP = ', TSTEP)
      call read_float(55, 'outfreq1 = ', outfreq1)
      call read_float(55, 'outfreq2 = ', outfreq2)
      call read_float(55, 'DELT = ', DELT)
      call read_int(55, 'KF = ', KF)              
      call read_float(55, 'ResidualThreshold = ', ResidualThreshold)
!
      call read_int(55, 'STATSFLAG = ', STATSFLAG)
      call read_int(55, 'RCFLAG = ', RCFLAG)
      call read_int(55, 'ICFLAG = ', ICFLAG)
      call read_int(55, 'WNFLAG = ', WNFLAG)
      call read_float(55, 'initial x shift = ', initialSHIFTX)
      call read_float(55, 'initial y shift = ', initialSHIFTY)
      call read_float(55, 'initial z shift = ', initialSHIFTZ)
      call read_int(55, 'lamFLAG = ', lamFLAG)
      call read_int(55, 'UPOflag = ', UPOflag)
!
      close(55)
!
      call params_update()
!
      end subroutine params_read_txt
!
!
!------------------------------------------------
!------------------------------------------------
! Read parameters from a simple text file
!
      subroutine paramsEXTRA_read_txt()
      implicit none
!
      open(55, FILE='paramsEXTRA.txt', ACTION='READ')
!
!     gmres specific parameters
      call read_int(55, 'IGUESSEND = ', IGUESSEND)
      call read_float(55, 'maxPERIOD = ', maxPERIOD)
      call read_float(55, 'gmres ResidualThreshold = ', gmresResT)
!
!     ALC specific parameters
      call read_logical(55, 'arclcFLAG = ', arclcFLAG)
      call read_logical(55, '1stOrder = ', FirstOrder)
      call read_logical(55, '2ndOrder PredOnly = ', SecondOrderPredOnly)
      call read_logical(55, 'Full 2ndOrder = ', FullSecondOrder)
      call read_logical(55, 'drN_reset = ',drN_reset)
      call read_logical(55, 'Fixed Period = ',FixedP)
      call read_float(55, 'Fixed Period Size = ',FixedPsize)
      call read_float(55, 'delta_v2 = ', delta_v2)
!
!     Arnoldi specific parameters
      call read_logical(55, 'ArnoldiFLAG = ', ArnoldiFLAG)

!
      close(55)
!
!     Initial Flags used by ALC algorithm (user should not adjust these)
      arclcFLAG2 = .FALSE.
      CONVERGED = .TRUE.
      FirstRun = .TRUE.
      SecondRun = .FALSE.
!
      end subroutine paramsEXTRA_read_txt

!
      end module parameters
!
