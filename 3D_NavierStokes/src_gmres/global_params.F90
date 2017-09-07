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
      INTEGER, PARAMETER  :: NZ = 128
      INTEGER, PARAMETER  :: NY = NZ
      INTEGER, PARAMETER  :: NX = INT(NZ/alpha)
      INTEGER, PARAMETER  :: NX2 = NX/2+1
!
      INTEGER, PARAMETER  :: KTZ = NZ/3
      INTEGER, PARAMETER  :: KTX = NX/3
      INTEGER, PARAMETER  :: KTY = NY/3
      INTEGER, PARAMETER  :: IKTX = KTX+1
      INTEGER, PARAMETER  :: IKTY = KTY+KTY+1
      INTEGER, PARAMETER  :: IKTZ = KTZ+KTZ+1

      INTEGER, PARAMETER  :: NR  = NY*NX*NZ 
      INTEGER, PARAMETER  :: NKT = IKTY*IKTX*IKTZ 

!
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
      INTEGER, DIMENSION(NKT) :: L  ! stencil for wavenumber plane
      LOGICAL, DIMENSION(NKT) :: LL  ! stencil for wavenumber plane
      LOGICAL, DIMENSION(3*NKT) :: L3  ! stencil for full state vector
      LOGICAL, DIMENSION(3*NKT) :: L2  ! stencil for full state vector
! ----------------------------------------------------------------------------
!  PHYSICAL PARAMETERS
!
      REAL(rk) ::  V2,AMPFOR,TSTEP,DELT,AMPV
      INTEGER  ::  KF,NSTOP,NOUT,NOUTV,OUTPLANE
!
!                                                                         
!$$$      INTEGER, PARAMETER  :: STORESIZE = 1000 !CHANGE 333 format() too
!$$$      INTEGER, PARAMETER  :: NTIMES = 100000 !CHANGE 333 format() too 
!                                                                         
      INTEGER, PARAMETER  :: GMRESMAX = 500                               
      INTEGER, PARAMETER  :: NEWTONMAX = 75                               
      INTEGER, PARAMETER  :: LWORKsvd = 7*GMRESMAX + 4*GMRESMAX*GMRESMAX  
      INTEGER, PARAMETER  :: LRWORKsvd= 7*GMRESMAX + 5*GMRESMAX*GMRESMAX  
!                                                                         
      INTEGER, PARAMETER  :: FFTW_EXHAUSTIVE = 8       !see fftw3.f       
      INTEGER, PARAMETER  :: FFTW_PATIENT = 32         !see fftw3.f       
      INTEGER, PARAMETER  :: FFTW_MEASURE = 0          !see fftw3.f       
      INTEGER, PARAMETER  :: FFTW_ESTIMATE = 64        !see fftw3.f       
      INTEGER, PARAMETER  :: FFTW_FLAG = FFTW_ESTIMATE                    

!
! ---------------------------------------------------------------------------
!  FFTs                                                                      
!                                                                            
      INTEGER(KIND=8) :: PlanK2XIR,PlanK2ETR,PlanK2ZTR,PlanK2UR,PlanK2VR,PlanK2WR
      INTEGER(KIND=8) ::PlanUR2K,PlanVR2K,PlanWR2K       
!                                                                            
! ----------------------------------------------------------------------------
!--------------------------------------------------------------------------------
!  FLAGS
!       
      INTEGER :: myISEED,ISEED,GRFLAG,TRFLAG,RCFLAG,STATSFLAG,ICFLAG,IREST
      INTEGER :: WNFLAG,SubSpaceFLAG,lamFLAG,UPOflag                   
!                                                                      
!--------------------------------------------------------------------------------
!  GMRES PARAMS                                                                  
!                                                                                
      INTEGER :: IGUESSEND                                                       
      REAL(rk) :: maxPERIOD,gmresResT,SNEWTX,SNEWTZ,SNEWTY
!
! --------------------------------------------------------------------------
!  COUNTERS
!
      INTEGER ::  izkout
!
!----------------------------------------------------------------------------
! TTGIRD
!
      REAL(rk)    :: ResidualThreshold
      INTEGER     :: NSHIFT
!
!$$$      INTEGER :: POINTER,ITIME
!$$$!
!$$$      REAL(KIND=4), DIMENSION(STORESIZE,NTIMES) :: TtGRID_BIGT,TtGRID_DZ
!$$$      REAL(rk),     DIMENSION
!  TIMING
!
      REAL(KIND=4) :: time1(2),time2(2),TIME3,T(2)
      REAL(KIND=4) :: TIC,TOC,MYTIME
!
! ---------------------------------------------------------------------------------
!  ALC
      REAL(rk) :: delta_v2,FixedPsize
      LOGICAL :: halveFLAG,doubleFLAG,changeDELT
      LOGICAL :: arclcFLAG,arclcFLAG2,CONVERGED
      LOGICAL :: FirstRun,SecondRun
      LOGICAL :: FirstOrder,FullSecondOrder,SecondOrderPredOnly
      LOGICAL :: drN_reset,FixedP
!
!-----------------------------------------------------------------------
! Arnoldi
      LOGICAL ::  ArnoldiFLAG
!
! ---------------------------------------------------------------------------------
!  SYMMETRIC SUBSPACE
!
      INTEGER :: vertex,rotateYFLAG
      REAL(rk) :: initialSHIFTX,initialSHIFTY,initialSHIFTZ
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
