!  
      module GLOBAL_PARAMS
      IMPLICIT NONE
!
! ------------------------------------------------------------------------
!     PARAMETERS
! ------------------------------------------------------------------------
!
      INTEGER, PARAMETER  :: rk = 8      ! floating point precision      
!
      REAL(rk), PARAMETER :: alpha = 1.0_rk
!
      INTEGER, PARAMETER  :: NY = 1024
      INTEGER, PARAMETER  :: NX = INT(NY/alpha)
!
      INTEGER, PARAMETER  :: KTX = NX/3
      INTEGER, PARAMETER  :: KTY = NY/3
      INTEGER, PARAMETER  :: IKTX = KTX+1
      INTEGER, PARAMETER  :: IKTY = KTY+KTY+1
!
      INTEGER, PARAMETER  :: STORESIZE = 1000 !CHANGE 333 format() too
      INTEGER, PARAMETER  :: NTIMES = 1000 !CHANGE 333 format() too
!
      INTEGER, PARAMETER  :: FFTW_EXHAUSTIVE = 8       !see fftw3.f
      INTEGER, PARAMETER  :: FFTW_PATIENT = 32         !see fftw3.f
      INTEGER, PARAMETER  :: FFTW_MEASURE = 0          !see fftw3.f
      INTEGER, PARAMETER  :: FFTW_ESTIMATE = 64        !see fftw3.f
      INTEGER, PARAMETER  :: FFTW_FLAG = FFTW_ESTIMATE
!
!  -----------------------------------------------------------------------
!  CONSTANTS
!
      COMPLEX(rk) :: ZI 
      REAL(rk)    :: TWOPI 
      REAL(rk)    :: ROOT2 
!
! ------------------------------------------------------------------------
!  MASK
! 
      INTEGER, DIMENSION(IKTX,IKTY) :: L  ! stencil for wavenumber plane
      LOGICAL, DIMENSION(IKTX,IKTY) :: LL  ! stencil for wavenumber plane
!
! ---------------------------------------------------------------------------
!  FFTs
!
      INTEGER(KIND=8) :: PlanK2ZR,PlanK2UR,PlanK2VR,PlanNZR2K,PlanUR2K
!
! ----------------------------------------------------------------------------
!  PHYSICAL PARAMETERS
!
      REAL(rk), DIMENSION(IKTX,IKTY) :: NU
      REAL(rk) ::  V1,V2,AMPFOR,TSTEP,DELT,AMPV,scale
      INTEGER  ::  KF,NSTOP,NOUT,NOUTV
!
!--------------------------------------------------------------------------------
!  FLAGS
!
      INTEGER :: ISEED,RCFLAG,STFLAG,ICFLAG,IREST
      INTEGER :: WNFLAG,SubSpaceFLAG,lamFLAG,UPOflag
!
! --------------------------------------------------------------------------
!  COUNTERS
!
      INTEGER ::  izkout
!----------------------------------------------------------------------------
! TTGIRD
!
      REAL(rk)    :: ResidualThreshold
!
      INTEGER, PARAMETER  :: NSHIFTY = 4
      INTEGER, PARAMETER  :: NSHIFTX = 30
      INTEGER :: POINTER,ITIME,TtOUTflag
!
      REAL(rk)    :: minRESIDUALouter,minRESIDUALouter2
      REAL(rk)    :: minTIMEouter,TtPERIOD,minSHIFTXouter,minSHIFTYouter
      COMPLEX(rk) :: minZouter(IKTX,IKTY)
!$$$      REAL(rk)    :: TtTimer,TtWinMax,TtTimerStart
!
! ------------------------------------------------------------------------------
!  SYMMETRIC SUBSPACE
!
      INTEGER :: vertex,rotateYFLAG
      REAL(rk) :: initialSHIFTX,initialSHIFTY
!
! ---------------------------------------------------------------------------------

! ------------------------------------------------------------------------------
!  TIMING
!
      REAL(KIND=4) :: time1(2),time2(2),TIME3,T(2)
      REAL(KIND=4) :: TIC,TOC,MYTIME
!
! ---------------------------------------------------------------------------------
!     INFILE
!
      character(len=256) :: infile
      character(len=256) :: InfileType
!
!  ---------------------------------------------------------------------------------
      end module GLOBAL_PARAMS
!
