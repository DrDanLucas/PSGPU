     PROGRAM MAIN 
!
      use GLOBAL_PARAMS
      use PARAMETERS
      use io_arpack
      IMPLICIT NONE 
!
      COMPLEX(rk), DIMENSION(NKT) :: XII,ETA,ZET
      COMPLEX(rk), DIMENSION(3*NKT) :: ZN,ZO,UK
      COMPLEX(rk), DIMENSION(3*NKT) :: ZNEWT,ZNN,ZN2,ZNSHIFT
      COMPLEX(rk), DIMENSION(3*NKT) :: DIFFZ,JFDIFF
      COMPLEX(rk), DIMENSION(3*NKT) :: Zplus,Zplus2,FNplus,FNplus2
      COMPLEX(rk), DIMENSION(3*NKT) :: FN,FNpred,FNpredC
      COMPLEX(rk), DIMENSION(3*NKT) :: dZNdt,dzdt,v
      COMPLEX(rk), DIMENSION(3*NKT) :: dZNdSx,dzdSx
      COMPLEX(rk), DIMENSION(3*NKT) :: dZrdr,dZrdrO2,d2Zrdr2
      COMPLEX(rk), DIMENSION(3*NKT) :: Zr0,Zr1,Zr2
      COMPLEX(rk),DIMENSION(:,:),ALLOCATABLE :: q
      COMPLEX(rk) :: FF2(NX2*NY*NZ)
      COMPLEX(rk) :: TERMZ,TZ,C1,AVZ,SHIFT
!
      REAL(rk), DIMENSION(NKT)      :: NUZN,NU1,NU,KX,KY,KZ
      REAL(rk), DIMENSION(NR)       :: UR,VR,WR,XIR,ETR,ZTR
      REAL(rk) :: G05DAF
      REAL(rk) :: RANNO,DMZ,DPZ,TSTART,TIME
      REAL(rk) :: r1,r2,d1,d2y,dq13,dq23,qamp,q1,q2,q3,xc,yc,x,y,rg
      REAL(rk) :: phi,s1,s2,qpert,S,r,amplitude,xp,yp,theta,cs,sn
      REAL(rk) :: kkx,kky,kkz,wk
      REAL(rk), DIMENSION(GMRESMAX+1,GMRESMAX) :: H,Hcopy
      REAL(rk), DIMENSION(GMRESMAX)            :: Ssvd,Xh
      REAL(rk), DIMENSION(GMRESMAX+1)          :: P,Yh,qN1,qN2,qN3
      REAL(rk) :: Usvd(GMRESMAX+1,GMRESMAX+1),VHsvd(GMRESMAX,GMRESMAX)
      REAL(rk) :: WORKsvd(LWORKsvd)
      REAL(rk) :: NER,NERold,GMR,const,FN1pred,FN2pred,FN3pred
      REAL(rk) :: NORMFN,NORMFNplus,NORMFNplus2,NORMFNpred,NORMFNpredC
      REAL(rk) :: NORMZ
      REAL(rk) :: NETOL,GMTOL,epsilon,DELTA,GMRESdelta
      REAL(rk) :: vN1,vN2,vN3,normX,MU1,MU2,MUmid
      REAL(rk) :: TNEWT,DIFFT,Tplus,Tplus2,TRUEDELT,TrueTRUEDELT
      REAL(rk) :: DIFFSx,Sxplus,Sxplus2
      REAL(rk) :: TSTARTin,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin,v2in
      REAL(rk) :: R3,Narc,Narcplus
      REAL(rk) :: delta_r1,delta_r2,Ndelta_r,Ndelta_rmax,drrat
      REAL(rk) :: NORMb,v2N,v2plus,v2plus2,DIFFv2,DIFFv3
      REAL(rk) :: Tr0,Tr1,Tr2,Sr0,Sr1,Sr2,v2r0,v2r1,v2r2
      REAL(rk) :: dTrdr,dSrdr,dv2rdr,dTrdrO2,dSrdrO2,dv2rdrO2
      REAL(rk) :: d2Trdr2,d2Srdr2,d2v2rdr2
      REAL(rk) :: NORMbplus,NORMbpredC
!

      INTEGER ::  IKF(NX2*NZ*NY), IKN(NKT)
      INTEGER ::  IKX,IKY,IKZ,jw,KFX1,KFX2,KFY1,KFY2,KFZ2,KFZ1,KFT1,KFT2
      INTEGER ::  NT,IKK,IK,IT,NERcount,IKT,IYY,IZZ
      INTEGER ::  i,j,ix,jx,m,ic,jc,biggest,id,jd,nvort
      INTEGER ::  INE,IGM,IORTH,INFOsvd,IWORKsvd(8*GMRESMAX)
      INTEGER ::  IGUESS,GUESSARRAY(2),iuposout,IGUESSSTART
      INTEGER ::  IHOOK,IHOOKMAX
      INTEGER ::  GMREScount,HOOKcount,AllocateStatus
      INTEGER ::  convergeFLAG
!
      EQUIVALENCE (XII,XIR), (ETA,ETR), (ZET,ZTR)
!
      integer(kind=4), parameter ::                                        &
    &       nev = 30,                                               &
    &       ncv = 120,                                               &
    &       lworkl = 3*ncv*ncv + 6*ncv
!
      real(rk), parameter :: TOL = 0.000001_rk
!
      integer(kind=4) :: ido, info, cnt, switch, outfreqarn
      character(len=2) :: which
!
      integer(kind=4), dimension(11) :: iparam
      integer(kind=4), dimension(14) :: ipntr
      logical, dimension(ncv) :: sel
!
      real(rk),DIMENSION(:),ALLOCATABLE :: resid,VarnC,workd
      real(rk),DIMENSION(:,:),ALLOCATABLE :: Varn,myeigvecs
      real(rk), dimension(lworkl) :: workl
!
      real(rk), dimension(3*ncv) :: workev
      real(rk), dimension(nev+1) :: DR, DI
      real(rk), dimension(ncv) :: myritzr,myritzi,myritzerr
      real(rk), dimension(ncv) :: lambdar,lambdai
      real(rk), dimension(ncv) :: WWR,WWI
      real(rk), dimension(5*ncv) :: dgeevWORK
      real(rk), dimension(ncv,ncv) :: Hess,dgeevVL,dgeevVR
      real(rk) :: lambdarsum     
!
      integer :: ArnoldiEXIT, Iarn, Iarn2, dgeevINFO
!
!--------------------------------------------------------------------------
! ARPACK trace output

      integer(kind=4)  logfil, ndigit, mgetv0, &
           msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd, &
           mnaupd, mnaup2, mnaitr, mneigh, mnapps, mngets, mneupd, &
           mcaupd, mcaup2, mcaitr, mceigh, mcapps, mcgets, mceupd
      common /debug/ &
           logfil, ndigit, mgetv0, &
           msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd, &
           mnaupd, mnaup2, mnaitr, mneigh, mnapps, mngets, mneupd, &
           mcaupd, mcaup2, mcaitr, mceigh, mcapps, mcgets, mceupd
      ndigit = -6
      mnaup2 = 2
      mnaitr = 3

      call set_gpu(0)

!-----------------------------------------------------------------------
!
      print*,'rk = ',rk
      print*,'Truncation wavenumber = ',(REAL(kty,rk)-0.5_rk)
!
! -------------------------------------------------------------------
!
!
!     Read in input parameters from file and set constants.
      CALL params_read_txt()   
      TNEWT = TSTEP   
      SNEWTX = 0._rk   
      SNEWTY = 0._rk
      SNEWTZ = 0._rk
      TrueTRUEDELT = DELT   
!
!
!     GMRES-HOOKSTEP PARAMETERS
!
      GUESSARRAY = (/ 4,14 /)
      NETOL  = 10._rk**(-10._rk)
      GMTOL  = 0.001_rk 
      epsilon = 10._rk**(-7._rk)
      GMRESdelta = 0.1_rk
      const = 0._rk  ! 10._rk**(-4._rk)
      IHOOKMAX = 50
      changeDELT = .FALSE.
      convergeFLAG = 0
!
      CALL paramsEXTRA_read_txt()
      NORMb = 1._rk
      NORMbplus = 0._rk

      ALLOCATE ( q(3*NKT,GMRESMAX+1),STAT=AllocateStatus)
      IF (AllocateStatus /= 0) STOP "*** Not enough memory for q ***"

      IF(ArnoldiFLAG) THEN
         ALLOCATE ( resid(SzV),STAT=AllocateStatus)
         IF (AllocateStatus /= 0) STOP "*** Not enough memory for resid ***"
         ALLOCATE ( VarnC(SzV),STAT=AllocateStatus)
         IF (AllocateStatus /= 0) STOP "*** Not enough memory for VarnC ***"
         ALLOCATE ( Varn(SzV,ncv),STAT=AllocateStatus)
         IF (AllocateStatus /= 0) STOP "*** Not enough memory for Varn ***"
         ALLOCATE ( myeigvecs(SzV,ncv),STAT=AllocateStatus)
         IF (AllocateStatus /= 0) STOP "*** Not enough memory for myeigvecs ***"
         ALLOCATE ( workd(3*SzV),STAT=AllocateStatus)
         IF (AllocateStatus /= 0) STOP "*** Not enough memory for workd ***"
      ENDIF
!
! -------------------------------------------------------------------------------
!
!     Create files for outputting UPO information
      open (150,file='UPOinfo.dat',form='formatted')                    
      WRITE(150,1501)   'Guess No : Start time : guess Re :guess Period &
     &:guess Shiftx: guess Shifty: guess Shiftz: guess Res  :      Status     : UPO N&
     &o :   Re   : Period : Shift x : Shift y : Shift z : Residual     : #NEWTO&
     &N : #GMRES : #HOOK '        
      open (999,file='UPOs.out',form='unformatted')
      write(999) 1,IKTX,IKTY,IKTZ  
      iuposout=0
!
!
! ---------------------------------------------------------------------------------------------
! -------------------------------------------------------------------------------------------- 
! --------------- Main Loop that cycles through different UPO guesses -----------------------    
! -------------------------------------------------------------------------------------------
! -------------------------------------------------------------------------------------------
!
!     Set starting guess number
      IGUESSSTART = IREST
      IF(arclcFLAG) IGUESSEND = IREST + IGUESSEND  ! IGUESSEND in this case is maximum number of tracing steps                 
!      IF(ArnoldiFLAG) IGUESSEND = IREST  ! Only 1 UPO is put through ARPACK
!
!     Start loop 
      DO IGUESS=IGUESSSTART,IGUESSEND
!
         IREST = IGUESS
!        Reset IREST for tracing                                                                                                      
         IF(arclcFLAG) THEN
            IF(FirstRun) THEN
               IREST = IGUESSSTART
            ELSE
               IREST = 0
            END IF
         END IF
!
!     Reset some paramters
         DELTA = GMRESdelta
         TRUEDELT = TrueTRUEDELT
!
! Most of the following standard initialization stuff is the same as in the DNS code
! ------------------------------------------------------------------------------------
!
! INITIALISE STUFF, GET THE FFT READY, ETC.
!
      TIME  = RANNO(myISEED) !Initializes random number generator 
!
!     Initialize forcing wavenumber indices
      IF (NOUT.EQ.0) NOUT = 1
      IF (NOUTV.EQ.0) NOUTV = 1
!      IF (AMPFOR.NE.0._rk) THEN
        KFX1 =  1
        KFX2 =  1
        KFY1 =  KF+1+KTY
        KFY2 =  KF+1+KTY
        KFZ1 = KTZ+1+KF
        KFZ2 = KTZ+1-KF
        KFT1 = ((KFZ1-1)*IKTY+KFY1-1)*IKTX+KFX1
        KFT2 = ((KFZ2-1)*IKTY+KFY1-1)*IKTX+KFX1
!        KFX1 =>  KX = KF
!        KFY1 =>  KY = 0
!        KFX2 =>  KX = 0
!        KFY2 =>  KY = KF
!      ENDIF
!
!  -----------------------------------------------------------------------
!    Setup output files
!      open (99,file='Zk.out',form='unformatted')
!      write(99) 1,IKTX,IKTY
!      open (13,file='spec.dat',form='formatted')
!      open (31,file='spec2D.dat',form='unformatted')
!      write(13,*) jw, ktx-1
!      if (trflag.eq.1) then
!        open (23,file='tran.dat',form='formatted')
!        write(23,*) jw, ktx-1
!      endif
!      open (15,file='engy.dat',form='formatted')
!      open (51,file='probe.dat',form='formatted',position='rewind')
!      write(15,*) jw
!
! ----------------------------------------------------------------------------
!     Create FFTW plans, one for each transform in convol.f
      call dfftw_plan_dft_c2r_3d(PlanK2XIR,NX,NY,NZ,FF2,XIR,FFTW_FLAG)
      call dfftw_plan_dft_c2r_3d(PlanK2ETR,NX,NY,NZ,FF2,ETR,FFTW_FLAG)
      call dfftw_plan_dft_c2r_3d(PlanK2ZTR,NX,NY,NZ,FF2,ZTR,FFTW_FLAG)
      call dfftw_plan_dft_c2r_3d(PlanK2UR,NX,NY,NZ,FF2,UR,FFTW_FLAG)
      call dfftw_plan_dft_c2r_3d(PlanK2VR,NX,NY,NZ,FF2,VR,FFTW_FLAG)
      call dfftw_plan_dft_c2r_3d(PlanK2WR,NX,NY,NZ,FF2,WR,FFTW_FLAG)
      call dfftw_plan_dft_r2c_3d(PlanUR2K,NX,NY,NZ,UR,FF2,FFTW_FLAG)
      call dfftw_plan_dft_r2c_3d(PlanVR2K,NX,NY,NZ,VR,FF2,FFTW_FLAG)
      call dfftw_plan_dft_r2c_3d(PlanWR2K,NX,NY,NZ,WR,FF2,FFTW_FLAG)
!
! -------------------------------------------------------------------------------
!     Initialize variables
      NT = 0
      MYTIME =   0._rk
      ZN  = CMPLX(0._rk,0._rk,rk)
      ZNN  = CMPLX(0._rk,0._rk,rk)
      ZN2  = CMPLX(0._rk,0._rk,rk) 
      ZNSHIFT  = CMPLX(0._rk,0._rk,rk)
      ZNEWT = CMPLX(0._rk,0._rk,rk)
      ZO  = CMPLX(0._rk,0._rk,rk)
      UK  = CMPLX(0._rk,0._rk,rk)
!
      DIFFT = 0._rk
      FN     = CMPLX(0._rk,0._rk,rk)
      FNplus = CMPLX(0._rk,0._rk,rk)
      DIFFZ  = CMPLX(0._rk,0._rk,rk)
      JFDIFF = CMPLX(0._rk,0._rk,rk)
      dZNdt  = CMPLX(0._rk,0._rk,rk)
      dzdt   = CMPLX(0._rk,0._rk,rk)
      dZNdsx  = CMPLX(0._rk,0._rk,rk)
      dzdsx   = CMPLX(0._rk,0._rk,rk)
      v      = CMPLX(0._rk,0._rk,rk)
      q      = CMPLX(0._rk,0._rk,rk)
      halveFLAG = .FALSE.
      doubleFLAG = .FALSE.
!
      TSTARTin = 0._rk 
      TNEWTin = FixedPsize
      SNEWTXin = 0._rk 
      SNEWTYin = 0._rk 
      SNEWTZin = 0._rk 
      NERin = 0._rk 
      NER = 0._rk 
      NERold = 0._rk
      NERcount = 0
!     
      if (irest.eq.0) TSTART=0._rk
! -----------------------------------------------------------------------------  
!   
!     Read in starting state and initialize arrays 
      CALL INIT (XII,ETA,ZET,KX,KY,KZ,NU,NUZN,NU1,TSTARTin,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin)

      normX = SQRT( SUM( REAL(XII*CONJG(XII)), LL ) + SUM( REAL(ETA*CONJG(ETA)), LL )  &
           & + SUM( REAL(ZET*CONJG(ZET)), LL ) )
      print*, 'normX = ',normX

      DO IKT = 1,NKT
               ZO(IKT)       = XII(IKT)
               ZO(NKT+IKT)   = ETA(IKT)
               ZO(2*NKT+IKT) = ZET(IKT)
      ENDDO
      NORMX = SQRT( SUM( REAL(ZO*CONJG(ZO))) )
      print*, 'normZ3 = ',normX

!      CALL SPEC (ZO,TSTARTin)
!      stop
      !     Store index array for padding before KR FFT
      DO IKX=1,IKTX
         DO IKY=1,KTY+1
            IYY = IKY-1+KTY
            DO IKZ=1,KTZ+1
               IZZ = IKZ-1+KTZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX

               IKF(IK) = IKX-1+IKTX*(IYY+IKTY*IZZ)
            ENDDO
            DO IKZ=KTZ+2,NZ-KTZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX
               IKF(IK) = -1
            ENDDO
            DO IKZ=NZ-KTZ+1,NZ
               IZZ = IKZ-1+KTZ-NZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX

               IKF(IK) = IKX-1+IKTX*(IYY+IKTY*IZZ)
            ENDDO
         ENDDO
         DO IKY=KTY+2,NY-KTY
            DO IKZ=1,NZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX
               IKF(IK) = -1
            ENDDO
         ENDDO
         DO IKY=NY-KTY+1,NY
            IYY = IKY-1+KTY-NY
            DO IKZ=1,KTZ+1
               IZZ = IKZ-1+KTZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX

               IKF(IK) = IKX-1+IKTX*(IYY+IKTY*IZZ)
            ENDDO
            DO IKZ=KTZ+2,NZ-KTZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX
               IKF(IK) = -1
            ENDDO
            DO IKZ=NZ-KTZ+1,NZ
               IZZ = IKZ-1+KTZ-NZ
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX

               IKF(IK) = IKX-1+IKTX*(IYY+IKTY*IZZ)
            ENDDO
         ENDDO
      ENDDO

!
      DO IKZ=1,NZ
         DO IKY=1,NY
            DO IKX=IKTX+1,NX/2+1
               IK = ((IKZ-1)*NY +IKY-1)*NX2 + IKX
               IKF(IK) = -1
            ENDDO
         ENDDO
      ENDDO

!     Store index array for rewriting and normalising after RK FFT
       DO IKX =1,IKTX
          DO IKZ =1,KTZ
             IZZ = NZ+IKZ-1-KTZ
             DO IKY =1,KTY
                IYY = NY+IKY-1-KTY

                IK  = ((IKZ-1)*IKTY+IKY-1)*IKTX +IKX
                IKK = (IZZ*NY+IYY)*NX2 + IKX-1

                IKN(IK) = IKK
             ENDDO
             DO IKY =KTY+1,IKTY
                IYY=IKY-1-KTY

                IK  = ((IKZ-1)*IKTY+IKY-1)*IKTX +IKX
                IKK = (IZZ*NY+IYY)*NX2 + IKX-1

                IKN(IK) = IKK
             ENDDO
          ENDDO
          DO IKZ =KTZ+1,IKTZ
             IZZ=IKZ-1-KTZ
             DO IKY =1,KTY
                IYY = NY+IKY-1-KTY

                IK  = ((IKZ-1)*IKTY+IKY-1)*IKTX +IKX
                IKK = (IZZ*NY+IYY)*NX2 + IKX-1

                IKN(IK) = IKK
             ENDDO
             DO IKY =KTY+1,IKTY
                IYY=IKY-1-KTY

                IK  = ((IKZ-1)*IKTY+IKY-1)*IKTX +IKX
                IKK = (IZZ*NY+IYY)*NX2 + IKX-1

                IKN(IK) = IKK
             ENDDO

          ENDDO
       ENDDO


!$$$      CALL INIT (ZO,TSTARTin)
!
! --------------------------------------------------------------------------------
!   *****************************************************************
!   *       GMRES Guess Filters                                      *
!   *****************************************************************
!
!     Decide whether or not to attempt guess 
      IF (UPOflag.eq.1) THEN
         Print*,'   '
         Print*,'PERIODin = ',TNEWTin
         Print*,'SHIFTXin = ',SNEWTXin
         Print*,'SHIFTYin = ',SNEWTYin
         Print*,'SHIFTZin = ',SNEWTZin
         Print*,'Residual = ',NERin
!
         IF( .NOT.(arclcFLAG .OR. ArnoldiFLAG)) THEN  ! only apply guess filters if alc and stability are off
!        *****************************************************************************
            IF (TNEWTin.GT.maxPERIOD .OR. NERin.GT.gmresResT) THEN
!   .OR.             &      
!     &    ( SNEWTXin.LT.0.001_rk .AND. SNEWTYin.LT.0.001_rk  ) ) THEN             
!        *****************************************************************************
               print*,'Not trying Guess number ',IREST
               WRITE(150,1502) IREST,'    ',                                 &
     &              TSTARTin,1._rk/v2,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin,   &
     &   ' :   "NOT TRIED"   : ',0,1._rk/v2,TNEWTin,SNEWTXin,SNEWTYin,   &
     &               NERin 
               CYCLE
            ENDIF   
         END IF
!        Initialize Guess variables
         TNEWT = TNEWTin
         SNEWTX = SNEWTXin
         SNEWTY = SNEWTYin
         SNEWTZ = SNEWTZin
         NER = NERin
         IF (FixedP) THEN
            SNEWTX = FixedPsize*SNEWTX/TNEWT
            TNEWT = FixedPsize
         END IF
      ENDIF
! -----------------------------------------------------------------------------------------
!
!     Setup Arnoldi variables
      IF(ArnoldiFLAG) THEN
         IF( SzV .NE. 2*COUNT(L3) ) THEN
            Print*,'Arnoldi error: SzV=',SzV,' is not equal to 2*COUNT(L3)=',2*COUNT(L3)
            STOP
         ENDIF
         ArnoldiEXIT = 0
!!$         call ic_apply(resid) 
         which = 'LM'       ! specifies which nev eigenvalues are to be computed.
!                           ! 'LM' = Eigenvalues of largest magnitude
!                           ! 'LR' = Eigenvalues of largest real part
         outfreqarn = 20
         cnt = 0
         ido = 0
         info = 0               ! ARPACK creates starting vector
!         info = 1               ! use supplied vector in "resid" as starting vector
         iparam(1) = 1          ! "exact" shifting
         iparam(3) = 1000       ! max number of implicit restarts
         iparam(7) = 1          ! mode
         switch = 0
!         resid = 1._rk
      ENDIF
!
! ----------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------
!                                                                                   
      IF(arclcFLAG) THEN
!                                                                                                                                   
!        ---------------------------------------------------------------------------                                                
!        --------------------------------------------------------------------------                                                 
         IF(FirstRun) THEN
!        -------------------------------------------------------------------------                                                  
!        --------------------------------------------------------------------------                                                 
!                                                                                                                                    
!           Do standard continuation step with fixed change in Re                                                                    
            IF(CONVERGED) THEN
               v2 = v2 - delta_v2
            ELSE
               v2 = v2 + delta_v2
               delta_v2 = delta_v2 / 2._rk
               v2 = v2 - delta_v2
            END IF
            WHERE (L3) Zr2 = ZO
            IF(.NOT.FixedP) Tr2 = TNEWT
            Sr2 = SNEWTX
            v2r2 = v2 + delta_v2
            CONVERGED = .FALSE.
!                                                                                                                                    
!        ----------------------------------------------------------------------------                                                
!        ----------------------------------------------------------------------------                                                
         ELSE
!        ---------------------------------------------------------------------------                                                
!        ----------------------------------------------------------------------------                                                
!                                                                                                                                    
!           Code should hopefully realize when UPOs are equilibrium points or T-Ws                                                   
            IF (Tr2 .LT. FixedPsize) THEN
               print*,'Period has decreased below',FixedPsize
               print*,'UPO is probably an equilibrium point or travellin&                                                            
     &g wave'
               print*,'Setting Fixed Period flag to .TRUE.'
               FixedP = .TRUE.
               Sr2 = FixedPsize*Sr2/Tr2
               Tr2 = FixedPsize
               Sr1 = FixedPsize*Sr1/Tr1
               Tr1 = FixedPsize
               Sr0 = FixedPsize*Sr0/Tr0
               Tr0 = FixedPsize
            END IF
!        ----------------------------------------------------------------------------                                               
!                                                                                                                                    
            Print*,'delta_r2 from previous step: ',delta_r2
!           calculate delta_r (ALC step size)                                                                                        
            delta_r2 = SQRT( SUM( REAL((Zr2-Zr1)*CONJG(Zr2-Zr1)), L3 )  &
     &            + (Sr2-Sr1)**2 + (v2r2-v2r1)**2 )
            IF(.NOT.FixedP) delta_r2 = SQRT(delta_r2**2 + (Tr2-Tr1)**2)
            Print*,'actual delta_r2: ',delta_r2
!                                                                                                                                   
!           approximate derivates wrt r                                                                                             
            WHERE (L3) dZrdr = (Zr2-Zr1)/delta_r2
            IF(.NOT.FixedP) dTrdr = (Tr2-Tr1)/delta_r2
            dSrdr = (Sr2-Sr1)/delta_r2
            dv2rdr = (v2r2-v2r1)/delta_r2
            Print*,'set 1st Order derivatives for correction step'
!        -------------------------------------------------------------------------                                                   
!                                                                                                                                    
            IF(SecondRun) THEN
!            set new delta_r for third run                                                                                           
               IF(CONVERGED) Ndelta_r = 2._rk*delta_r2
               IF(.NOT.CONVERGED) Ndelta_r = Ndelta_r/2._rk
               Ndelta_rmax = Ndelta_r
            ELSE
               IF(CONVERGED) THEN
                  IF(delta_r2 .LT. Ndelta_rmax) THEN
                     Ndelta_r = delta_r2
                  ELSE
                     Ndelta_r = Ndelta_rmax
                  END IF
                  IF(1.1_rk*Ndelta_r .LT. Ndelta_rmax) THEN
                     Ndelta_r = 1.1_rk*Ndelta_r
                  END IF
               END IF
               IF(.NOT.CONVERGED) Ndelta_r = Ndelta_r/2._rk
            END IF
            CONVERGED = .FALSE.
!
!           ---------------------------------------------------------------------------                                              
            IF(SecondRun .OR. FirstOrder) THEN
!           -----------------------------------------------------------------------                                                  
!              DO first Order Prediction of variables                                                                                
!                                                                                                                                    
!              Extrapolate starting state                                                                                            
               WHERE (L3) ZO = Zr2 + dZrdr*Ndelta_r
               IF(FixedP)  THEN
                  TNEWT = FixedPsize
               ELSE
                  TNEWT = Tr2  + dTrdr*Ndelta_r
               END IF
               SNEWTX = Sr2  + dSrdr*Ndelta_r
               v2    = v2r2 + dv2rdr*Ndelta_r
!                                                                                                                                    
               Print*,'done 1st Order prediction'
!                                                                                                                                    
!           ---------------------------------------------------------------------------                                             
            ELSE
!           ------------------------------------------------------------------------                                                
!                                                                                                                                    
!              Do second order prediction of variables                                                                              
!                                                                                                                                   
               drrat = delta_r2 / (delta_r2 + delta_r1)
!                                                                                                                                   
!            approximate first derivatives                                                                                          
             WHERE (L3)                                                   &
     &     dZrdrO2 = ( (1._rk-drrat*drrat)*Zr2 - Zr1 + drrat*drrat*Zr0 )  &
     &               / ( delta_r2 * (1._rk-drrat) )
           IF(.NOT.FixedP) dTrdrO2 = ( (1._rk-drrat*drrat)*Tr2 - Tr1      &
     &   + drrat*drrat*Tr0 ) / ( delta_r2 * (1._rk-drrat) )
           dSrdrO2 = ( (1._rk-drrat*drrat)*Sr2 - Sr1 + drrat*drrat*Sr0 )  &
     &               / ( delta_r2 * (1._rk-drrat) )
         dv2rdrO2 = ( (1._rk-drrat*drrat)*v2r2 -v2r1 +drrat*drrat*v2r0 )  &
     &               / ( delta_r2 * (1._rk-drrat) )
!
!           ---------------------------------------------------------------------------                                             
!            approximate second derivatives                                                                                         
               WHERE (L3)                                               &
     &         d2Zrdr2 = ( (drrat-1._rk)*Zr2 + Zr1 - drrat*Zr0 )        &
     &               / ( 0.5_rk*delta_r2*delta_r2*(1._rk-1._rk/drrat) )
      IF(.NOT.FixedP) d2Trdr2 = ( (drrat-1._rk)*Tr2 + Tr1 - drrat*Tr0 ) &
     &               / ( 0.5_rk*delta_r2*delta_r2*(1._rk-1._rk/drrat) )
               d2Srdr2 = ( (drrat-1._rk)*Sr2 + Sr1 - drrat*Sr0 )        &
     &               / ( 0.5_rk*delta_r2*delta_r2*(1._rk-1._rk/drrat) )
               d2v2rdr2 = ( (drrat-1._rk)*v2r2 + v2r1 - drrat*v2r0 )    &
     &               / ( 0.5_rk*delta_r2*delta_r2*(1._rk-1._rk/drrat) )
!            -------------------------------------------------------------------------                                              
!                                                                                                                                   
!     Extrapolate starting state                                                                                                    
               WHERE (L3) ZO = Zr2 + dZrdrO2*Ndelta_r                     &
     &                             + 0.5_rk*Ndelta_r*Ndelta_r*d2Zrdr2
               IF(FixedP) THEN
                  TNEWT = FixedPsize
               ELSE
                  TNEWT = Tr2  + dTrdrO2*Ndelta_r                         &
     &                      + 0.5_rk*Ndelta_r*Ndelta_r*d2Trdr2
               END IF
               SNEWTX = Sr2  + dSrdrO2*Ndelta_r                            &
     &                      + 0.5_rk*Ndelta_r*Ndelta_r*d2Srdr2
               v2    = v2r2 + dv2rdrO2*Ndelta_r                           &
     &                      + 0.5_rk*Ndelta_r*Ndelta_r*d2v2rdr2
!                                                                                                                                   
               Print*,'done 2nd Order prediction'
!                                                                                                                                   
               IF(FullSecondOrder) THEN
!                 Reset derivatives for correction step using 2nd Order calcs                                                       
                  WHERE (L3) dZrdr =  dZrdrO2
                  IF(.NOT.FixedP) dTrdr = dTrdrO2
                  dSrdr =  dSrdrO2
                  dv2rdr = dv2rdrO2
             Print*,'reset derivatives for correction step to 2nd Order'
               END IF
!                                                                                                                                    
!           ---------------------------------------------------------------------------                                             
            END IF
!           ---------------------------------------------------------------------------
!                                                                                                                                   
!           If requested recalculate ALC step size based on new predicted values                                                  
            Print*,'Ndelta_r = ',Ndelta_r
            IF (drN_reset) THEN
              Ndelta_r = SQRT( SUM( REAL((ZO-Zr2)*CONJG(ZO-Zr2)), L3 )  &
     &            + (SNEWTX-Sr2)**2 + (v2-v2r2)**2 )
           IF(.NOT.FixedP) Ndelta_r = SQRT(Ndelta_r**2 + (TNEWT-Tr2)**2)
              Print*,'Reset Ndelta_r to: ',Ndelta_r
            END IF
!        ------------------------------------------------------------------------                                                    
!           Reset GMRES variables                                                                                                    
            TNEWTin = TNEWT
            SNEWTXin = SNEWTX
!        ------------------------------------------------------------------------                                                    
!                                                                                                                                    
!           Calculate initial Narc                                                                                                   
            Narc = SUM( REAL(dZrdr*CONJG(ZO-Zr2)), L3 )                 &
     &     + dSrdr*(SNEWTX-Sr2) + dv2rdr*(v2-v2r2)                       &
     &    - Ndelta_r
            IF(.NOT.FixedP) Narc = Narc + dTrdr*(TNEWT-Tr2)
            print*,'initial Narc = ',Narc
!                                                                                                                                    
!        -----------------------------------------------------------------------                                                     
!        ----------------------------------------------------------------------                                                      
         END IF
!        -----------------------------------------------------------------------                                                     
!        -----------------------------------------------------------------------                                                     
!                                                                                                                                    
!        update NU                                                                                                                  
         DO IKT = 1, NKT
            IF(.NOT.LL(IKT))     CYCLE
            WK = KX(IKT)**2+KY(IKT)**2+KZ(IKT)**2
            R3 = log10(v2)+log10(WK)
            NU(IKT)     =  10._rk**R3
         ENDDO
!                                                                                                                                    
      END IF
!              
! ------------------------------------------------------------------------------------------
! ------------------------------------------------------------------------------------------
! ------------------------------------------------------------------------------------------
!
!     Output to Screen
      Print*,'   '
      print*,'        alpha=',alpha
      print*,'           Re=',1._rk/v2
      print*,'          ktx=',ktx
      print*,'          kty=',kty
      print*,'          ktz=',ktz
      print*,'          NSTOP=',NSTOP
      print*,'          NOUT=',NOUT
      print*,'          NOUTV=',NOUTV
      print*,'       epsilon =',epsilon
      print*,'         delta =',delta
!      print*,'v1+v2*ktx**16=',v1+                                       &
!     &      10._rk**(log10(v2)+16._rk*log10(real(ktx,rk)))
!

      IF (SubSpaceFLAG.EQ.1) THEN
         Print*,'Working in symmetric sub-space'
      ELSE 
         Print*,'Working in full solution space'
      ENDIF
!
      CALL FLUSH(6)
!
!
      print*,'Number of degrees of freedom = ',2*COUNT(L3)
      print*,'Max coefficient of initial condition = ',                 &
     &                                               MAXVAL(ABS(ZO),L3)
!  ---------------------------------------------------------------------------
!
!     OUTPUT INITIAL STATE 
!
      TIME = TNEWT
!      write(99) 0._rk,v2,TNEWT,SNEWTX,SNEWTY,SNEWTZ,NER,                             &
!     &                         (ZO(IKT),IKT=1,3*NKT)
      izkout = 0
!      print*,'Wrote to Zk.out number ',izkout
      print*,'Period = ',TIME,'X shift =',SNEWTX,'Z shift =',SNEWTZ
!
!!
      if (icflag.eq.1) stop
!
!     -----------------------------------------------------------------------
      CALL etime(time2,tic)
!
      CALL FLUSH(6)
!
!     ------------------------------------------------------------------------
!     ------------------------------------------------------------------------
!
!     Calculate UPO and check error
!
!        If this is a guess from DNS then, due to numerical errors
!        the period stored may not be precisely the period used in the DNS.
!        The following 'IF' statement corrects for this, whilst hopefully
!        ignoring any partly converged guesses.
      DELT = TRUEDELT
      IF (ABS(TNEWT-NINT(TNEWT/DELT)*DELT).LT.10._rk**(-10._rk)) THEN
         print*,'Guess period is assumed to be from DNS'
         print*,'Adjusting period so it is exactly equivalent to DNS'
         TNEWT = NINT(TNEWT/DELT)*DELT
      ENDIF
!

!        Set ZNEWT = ZO (ZO gets updated during timestepping)
         WHERE (L3) ZNEWT = ZO  
         v2N = v2
!        From here onwards v2N mirrors ZNEWT and v2 mirrors Z0, i.e. v2N and ZNEWT
!        store the information for the current Newton step, whereas v2 and Z0 are 
!        temporary work variables that are used directly by the timestepping routine.
!
!      -------------------------------------------------------------------------
!        Timestep once round orbit
         v2in = v2
         TSTART = 0._rk
         DELT = TRUEDELT
         NSTOP  =   INT(TNEWT/DELT)  !Note this rounds down
         print*,'Calculating UPO'
         Print*,'Period = ',TNEWT,', dt = ',DELT,', NSTOP = ',NSTOP
         IF(arclcFLAG) Print*,'Re = ',1._rk / v2N

         DO IKT =1,NKT
            IF (LL(IKT))THEN
               NUZN(IKT) = (1._rk - 0.5_rk*NU(IKT)*DELT) / DELT
               NU1(IKT) = DELT / (1._rk + 0.5_rk*NU(IKT)*DELT)
               
               XII(IKT) = ZO(IKT)
               ETA(IKT) = ZO(NKT+IKT)
               ZET(IKT) = ZO(2*NKT+IKT)
            ENDIF
         ENDDO
!
         CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);

 !        
 !       Now do a shortened last time step to adjust period to TNEWT 
         TSTART = NSTOP*DELT
         DELT = TNEWT - NSTOP*DELT
!         print*,DELT

         IF (DELT.NE.0._rk) THEN
            WHERE (LL)
               NUZN = (1._rk - 0.5_rk*NU*DELT) / DELT
               NU1 = DELT / (1._rk + 0.5_rk*NU*DELT)
            END WHERE
            print*,'Calculating last time step of length: ',DELT
            NSTOP = 1
            CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);
            
         END IF
!     ----------------------------------------------------------------------------
!
!        Store final Z state 
         DO IKT=1,NKT
            IF(LL(IKT))THEN
               ZN(IKT)       = XII(IKT)
               ZN(NKT+IKT)   = ETA(IKT)
               ZN(2*NKT+IKT) = ZET(IKT)
               ZNN(IKT)       = XII(IKT)
               ZNN(NKT+IKT)   = ETA(IKT)
               ZNN(2*NKT+IKT) = ZET(IKT)
            ENDIF
         ENDDO
!
         !        Shift final state back in x-direction
         DO IKT = 1, NKT

                  IF(.NOT.LL(IKT))     CYCLE
                  SHIFT = EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ)*EXP(-ZI*KX(IKT)*SNEWTX)
                  ZNSHIFT(IKT) = SHIFT*ZN(IKT)
                  ZNSHIFT(NKT+IKT) = SHIFT*ZN(NKT+IKT)
                  ZNSHIFT(2*NKT+IKT) = SHIFT*ZN(2*NKT+IKT)
         ENDDO
!     -------------------------------------------------------------------------------     
!!
         NORMZ = SQRT( SUM( REAL(ZN*CONJG(ZN)), L3 ) )

!    -------------------------------------------------------------------------------
!      Calculate 1st residual
!        Calculate F = Z(z,T) - z
         WHERE (L3) FN = ZNSHIFT - ZNEWT       
!
         NORMFN = SQRT( SUM( REAL(FN*CONJG(FN)), L3 ) )
!        Use NormFN to check for convergence 
         NER = NORMFN/NORMZ
        
!    ---------------------------------------------------------------------------------
!
         Print*,'-----------------------------------------------------'
         PRINT*,'Newton convergence check: ||F|| / ||Z|| = ',NER
         Print*,'-----------------------------------------------------'
         Print*,'-----------------------------------------------------'
!
!        Check if calculated residuals match the ones from timestepping
!        These should be an exact match, if everything is correct!
         IF (UPOflag.eq.1) THEN
            IF (NER.ne.NERin) THEN
               print*,'WARNING: calculated NER does not ',              &
     &                       'exactly match NER from Zk.in'
               Print*,'Setting starting NER to calculated NER'
               NERin = NER
            ENDIF
         ENDIF
!  -----------------------------------------------------------------------------
!
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! ----------------------------------------------------------------------------
!     NEWTON ITERATIONS
! ----------------------------------------------------------------------------    
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!         
      GMREScount = 0
      HOOKcount = 0
      CALL CPU_TIME(tic)

      DO INE=1,NEWTONMAX
!
!        qN1 and qN2 are zero on first GMRES iteration 
!        and so do not have to be added to NORMFN
         NORMb = NORMFN
!        But qN3 is not always zero!                                                                                                
         IF(arclcFLAG2) NORMb = SQRT(NORMFN*NORMFN + Narc*Narc)
!
!
         ! SET q(1) = b/|b| (in this case b = - FN )
         WHERE (L3) q(:,1) = - FN/NORMb  
         qN1(1) = 0._rk
         qN2(1) = 0._rk
         IF(arclcFLAG2) qN3(1) = - Narc/NORMb
!
         IF(.NOT.FixedP) THEN
!        Calculate dZN/dt before GMRES iteration
            CALL TIMEDERIV(ZNSHIFT,dZNdt,                                       &
     &                  KFT1,KFT2,FF2,UK,UR,VR,WR,XIR,ETR,ZTR,KX,KY,KZ,NU,IKF,IKN)
!
!        Calculate dz/dt before GMRES iteration
            CALL TIMEDERIV(ZNEWT,dzdt,                                     &
     &                  KFT1,KFT2,FF2,UK,UR,VR,WR,XIR,ETR,ZTR,KX,KY,KZ,NU,IKF,IKN)
         END IF
!
!        Calculate dZN/dS and dz/dS before GMRES iteration
         DO IKT = 1, NKT
            IF(.NOT.LL(IKT))     CYCLE
            dZNdSx(IKT) = -ZI*KX(IKT)*ZNSHIFT(IKT)
            dzdSx(IKT) = -ZI*KX(IKT)*ZNEWT(IKT)

            dZNdSx(NKT+IKT) = -ZI*KX(IKT)*ZNSHIFT(NKT+IKT)
            dzdSx(NKT+IKT) = -ZI*KX(IKT)*ZNEWT(NKT+IKT)

            dZNdSx(2*NKT+IKT) = -ZI*KX(IKT)*ZNSHIFT(2*NKT+IKT)
            dzdSx(2*NKT+IKT) = -ZI*KX(IKT)*ZNEWT(2*NKT+IKT)

         ENDDO
!
!     Initialize GMRES Hessenberg matrix to zeros
         H = 0._rk 
         Hcopy = 0._rk 
         Ssvd = 0._rk
         Usvd = 0._rk
         VHsvd = 0._rk
         WORKsvd = 0._rk
         INFOsvd = 0
!
         Print*,'Starting new Newton step and GMRES Hookstep iteration.'
         Print*,'This is Newton step number', INE
!
!    __________________________________________________________________________________
!    --------------------------------------------------------------------------------
!        GMRES
!    _________________________________________________________________________________
!    ----------------------------------------------------------------------------------
!       
!        GMRES LOOP to solve Az=b
         DO IGM=1,GMRESMAX
            CALL FLUSH(6)
!
!      ********************************************************************************
!       All  ARPACK related stuff happens here and exits the Newton loops on completion 
!      *********************************************************************************            
            IF(ArnoldiFLAG) THEN
!
!             ===================================================================
               IF(ArnoldiEXIT .EQ. 0) THEN
!             ===================================================================
!               Do regular arnoldi step
! 
!     !$               if (cnt/outfreqarn*outfreqarn == cnt) then
!!$                  call arpack_state_write(ido,SzV,nev,TOL,resid,ncv,Varn,   &
!!$     &      SzV, iparam, ipntr, workd, workl, lworkl, info, cnt, switch)
!!$                  write(*,*) 'Wrote ARPACK state ', switch
!!$                  
!!$                  switch = switch + 1
!!$                  if (switch == 2) then 
!!$                     switch = 0
!!$                  endif
!!$               endif
        
                  cnt = cnt + 1
                  write(*,*) "cnt = ", cnt

                  call dnaupd(ido, 'I', SzV, which, nev, TOL, resid,       &
     &    ncv, Varn, SzV, iparam, ipntr, workd, workl, lworkl, info)
!
!                 Check output from dnaupd to see what to do next
                  if (ido .eq. -1 .or. ido .eq. 1) then
!                    carry on arnoldi iteration
                     call workd2q(workd(ipntr(1)), q(:,IGM))
!
                  else
!                    ARPACK is finished - start output process
                     print*,'ido =',ido
!                     open (532,file='arpackfinal.out',form='unformatted')
!                     write(532) nev, ncv, Varn, ipntr, workl
!                     close (532)
!
!                    This is an attempt to extract the eigenvectors using Lapacks DGEEV.
!                    This is useful when the number of converged eigenvalues (iparam(5))
!                    doesn't equal nev, which causes dneupd to fail.
!                    It currently gets the eigenvalues OK, but also DGEEV sometimes fails 
!                    and there is also something wrong with the 
!                    eigenvectors or the matrix multiplication with the arnoldi basis
!                    vectors because they don't behave as expected when substituted 
!                    back into the timestepping routine.
!                
!$$$                     Hess = RESHAPE( workl(ipntr(5):ipntr(5)+ncv*ncv-1), &
!$$$     &                                                  (/ ncv, ncv /) )
!$$$!                     Hess(3,1) = 0._rk  ! This is a correction to an error in ARPACK 
!$$$!                     
!$$$!$$$!                 Test matrix for dgeev   
!$$$!$$$                     DO Iarn=1,4
!$$$!$$$                        DO Iarn2=1,4
!$$$!$$$                           Hess(Iarn,Iarn2) = Iarn2 + 4*(Iarn-1)
!$$$!$$$                        ENDDO
!$$$!$$$                     ENDDO
!$$$                     Print*,Hess
!$$$!
!$$$                     CALL DGEEV('N','V',ncv,Hess,ncv,WR,WI,dgeevVL,ncv,  & 
!$$$     &                 dgeevVR,ncv, dgeevWORK, 5*ncv, dgeevINFO )
!$$$!
!$$$                     Print*,'dgeevINFO = ',dgeevINFO
!$$$                     Print*,'WR = ',WR
!$$$                     Print*,'WI = ',WI
!$$$                     Print*,dgeevVR
!$$$!
!$$$                     myeigvecs = matmul(Varn,dgeevVR)
!
                     myritzr = workl(ipntr(6):ipntr(6)+ncv-1)
                     myritzi = workl(ipntr(7):ipntr(7)+ncv-1)                    
                     myritzerr = workl(ipntr(8):ipntr(8)+ncv-1)
                     lambdar = LOG(SQRT(myritzr**2 + myritzi**2))/TNEWT
                     lambdai = ATAN2(myritzi,myritzr)/TNEWT
!
                     lambdarsum = 0._rk
                     Iarn2 = 0
                     open (222,file='eigenvalues.dat',form='formatted',access='APPEND')  
                     DO Iarn=1,ncv
                        IF (myritzerr(Iarn) .LT. 100._rk*TOL) THEN
                           IF (lambdar(Iarn) .GT. 0._rk) THEN 
                             Iarn2 = Iarn2 + 1
                             write(222,2221) Iarn2,lambdar(Iarn),lambdai(Iarn)
                             lambdarsum = lambdarsum + lambdar(Iarn)
                           END IF
                        ENDIF
                     END DO
                     write(222,2222) '     ' 
                     write(222,2221) IREST,lambdarsum,TNEWT
                     close (222)
 2221                FORMAT(I3,1x,F13.5,F13.5)
 2222                FORMAT(A)
!
!
         call arpack_final_output(info,nev,ncv,which,iparam,sel, DR, DI,     &
     &   Varn, SzV, workev, TOL,resid, ipntr, workd, workl,lworkl,TNEWT)
                     IF (iparam(5) .NE. nev) EXIT
!                 Check leading eigenvalue/eigenvector      
                     DO Iarn=1,nev
                      call workd2q(Varn(:,Iarn), q(:,IGM+Iarn-1))               
                     END DO
                     Iarn = 1
                     ArnoldiEXIT = IGM+nev
!
                  endif
!
!            ========================================================================
               ELSE
!            ========================================================================
!               DO stability output routines and exit Newton iterations
!
                  IF (DR(Iarn).EQ.DR(Iarn+1)   .AND.                         &
     &                DI(Iarn).EQ.-DI(Iarn+1)) THEN
                     VarnC = VarnC -                                      & 
     &                 (DR(Iarn)*Varn(:,Iarn) - DI(Iarn)*Varn(:,Iarn+1))
                     Print*,'Checking Real part of eigenvector ',Iarn
                     Print*,' A x_r - ( mu_r x_r - mu_i x_i ) ' 
                  ELSE IF (DR(Iarn).EQ.DR(Iarn-1)   .AND.                   &
     &                     DI(Iarn).EQ.-DI(Iarn-1)) THEN
                     VarnC = VarnC -                                    & 
     &             (DI(Iarn-1)*Varn(:,Iarn-1) + DR(Iarn-1)*Varn(:,Iarn))
                     Print*,'Checking Imag part of eigenvector ',Iarn-1
                     Print*,' A x_i - ( mu_i x_r + mu_r x_i ) '
                  ELSE
                     VarnC = VarnC - DR(Iarn)*Varn(:,Iarn)
                     Print*,'Checking eigenvector ',Iarn
                     Print*,' A x_r -  mu_r x_r ' 
                  END IF
                  Print*,'Norm of error vector: ',SQRT(SUM(VarnC*VarnC))
                  Print*,'error vector max component: ',                &
     &                 SQRT(MAXVAL(VarnC*VarnC))
                  Iarn = Iarn + 1
                  IF (IGM .EQ. ArnoldiEXIT) EXIT
!
!             ===================================================================
               END IF
!            ====================================================================
!
            ENDIF
!
!     **************************************************************************************
!         Main ARPACK Stuff ends here. Regular loops carry out Aq using existing code below 
!     **************************************************************************************
!
!     --------------------------------------------------------------------------------------
!         CALCULATE Aq
!     -------------------------------------------------------------------------------------
!
!           Calculate the main part of Aq
            WHERE (L3) ZO =  ZNEWT + epsilon*q(:,IGM)
            IF(arclcFLAG2) THEN
               v2 = v2N + epsilon*qN3(IGM)
               !        update NU 
               DO IKT = 1, NKT
                  IF(.NOT.LL(IKT))     CYCLE
                  WK = SQRT(KX(IKT)**2+KY(IKT)**2+KZ(IKT)**2)
                  R3 = log10(v2)+2._rk*log10(WK)
                  NU(IKT)     =  10._rk**R3
               ENDDO
            ENDIF
!
!$$$            Print*,'Check epsilon:                                          &
!$$$     &          ||epsilon*q(:,:,IGM)|| / || ZNEWT || = ',                   &
!$$$     &  SQRT( SUM( REAL(epsilon*q(:,:,IGM)*CONJG(epsilon*q(:,:,IGM))) ))
!$$$     &  / SQRT( SUM( REAL(ZNEWT*CONJG(ZNEWT)) ))
!
            TSTART = 0._rk
            DELT = TRUEDELT
            NSTOP  =   INT(TNEWT/DELT) !Note this rounds down

            DO IKT =1,NKT
               IF (LL(IKT))THEN
                  NUZN(IKT) = (1._rk - 0.5_rk*NU(IKT)*DELT) / DELT
                  NU1(IKT) = DELT / (1._rk + 0.5_rk*NU(IKT)*DELT)
                  
                  XII(IKT) = ZO(IKT)
                  ETA(IKT) = ZO(NKT+IKT)
                  ZET(IKT) = ZO(2*NKT+IKT)
               ENDIF
            ENDDO

            CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);
!        
!       Now do a shortened last time step to adjust period to TNEWT 
            TSTART = NSTOP*DELT
            DELT = TNEWT - NSTOP*DELT
            IF (DELT.NE.0._rk) THEN
               WHERE (LL)
                  NUZN = (1._rk - 0.5_rk*NU*DELT) / DELT
                  NU1 = DELT / (1._rk + 0.5_rk*NU*DELT)
               END WHERE
               NSTOP = 1
               CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);

            END IF

!        Store final Z state 
         DO IKT=1,NKT
            IF(LL(IKT))THEN
               ZN(IKT)       = XII(IKT)
               ZN(NKT+IKT)   = ETA(IKT)
               ZN(2*NKT+IKT) = ZET(IKT)
            ENDIF
         ENDDO

            
!    -------------------------------------------------------------------------------------
!
!     ************************************************************************************
!     That's all ARPACK needs from the GMRES routines
!      so do some extra Stability stuff and skip rest of GMRES loop      
            IF(ArnoldiFLAG) THEN
               DO IKT = 1, NKT
                  IF(.NOT.LL(IKT))     CYCLE
                  SHIFT = EXP(-ZI*KX(IKT)*SNEWTX)*EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ)
                  v(IKT) =SHIFT*( ZN(IKT)-ZNN(IKT) )/epsilon
                  v(NKT+IKT) =SHIFT*( ZN(NKT+IKT)-ZNN(NKT+IKT) )/epsilon
                  v(2*NKT+IKT) =SHIFT*( ZN(2*NKT+IKT)-ZNN(2*NKT+IKT) )/epsilon
               ENDDO
               IF(ArnoldiEXIT .EQ. 0) THEN
                  call v2workd(v, workd(ipntr(2)))
               ELSE
                  call v2workd(v, VarnC)
               ENDIF
               CYCLE
            END IF
!     *************************************************************************************
!
!   ------------------------------------------------------------------------------------

!     Calculate rows 1 to N of vector v in GMRES
            DO IKT = 1, NKT
               IF(.NOT.LL(IKT))     CYCLE
               SHIFT = EXP(-ZI*KX(IKT)*SNEWTX)*EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ) 
               DO IT = 1,3
                  IK = (IT-1)*NKT +IKT
                  v(IK) = SHIFT*( ZN(IK)-ZNN(IK) )/epsilon - q(IK,IGM) &
                       &                                         + dZNdSx(IK)*qN2(IGM)
                  IF(.NOT.FixedP) v(IK) = v(IK) + dZNdt(IK)*qN1(IGM)
               ENDDO
            ENDDO
!
!           Calculate v(N+1), v(N+2) and  v(N+3)      
            IF(.NOT.FixedP) vN1 = SUM( REAL(CONJG(dzdt)*q(:,IGM)),L3 )
            vN2 = SUM( REAL(CONJG(dzdSx)*q(:,IGM)),L3 )
!
!           If arc-length continuation calculate v(N+3)                                                                              
            IF(arclcFLAG2) THEN
               vN3 = SUM( REAL(CONJG(dZrdr)*q(:,IGM)),L3 )            &
     &           + dTrdr*qN1(IGM) + dSrdr*qN2(IGM) + dv2rdr*qN3(IGM)
            ENDIF
!   ------------------------------------------------------------------------------------
!
!           Carry out Orthogonalization of v
            DO IORTH=1,IGM
              H(IORTH,IGM) = SUM( REAL(CONJG(q(:,IORTH)) * v ),L3 ) 
             IF(.NOT.FixedP) H(IORTH,IGM) = H(IORTH,IGM) +qN1(IORTH)*vN1
                             H(IORTH,IGM) = H(IORTH,IGM) +qN2(IORTH)*vN2
             IF(arclcFLAG2) H(IORTH,IGM) = H(IORTH,IGM) +qN3(IORTH)*vN3
!
              WHERE (L3)        v = v   - H(IORTH,IGM)*q(:,IORTH)
              IF(.NOT.FixedP) vN1 = vN1 - H(IORTH,IGM)*qN1(IORTH)
                              vN2 = vN2 - H(IORTH,IGM)*qN2(IORTH)
              IF(arclcFLAG2)  vN3 = vN3 - H(IORTH,IGM)*qN3(IORTH)
             ENDDO
!   -----------------------------------------------------------------------------------
!
!        This part calculates: H(IGM+1,IGM) =  ||[ v(:,:) vN1 ] ||
            H(IGM+1,IGM) =  SUM( REAL(v*CONJG(v)),L3 )
            IF(.NOT.FixedP) H(IGM+1,IGM) = H(IGM+1,IGM) + vN1*vN1 
                            H(IGM+1,IGM) = H(IGM+1,IGM) + vN2*vN2
            IF(arclcFLAG2)  H(IGM+1,IGM) = H(IGM+1,IGM) + vN3*vN3
            H(IGM+1,IGM) =  SQRT(H(IGM+1,IGM))
!  -----------------------------------------------------------------------------------
!
!           Set vector for next GMRES step
            WHERE (L3) q(:,IGM+1) = v/H(IGM+1,IGM) 
            IF(.NOT.FixedP) qN1(IGM+1) = vN1/H(IGM+1,IGM) 
                            qN2(IGM+1) = vN2/H(IGM+1,IGM) 
            IF(arclcFLAG2)  qN3(IGM+1) = vN3/H(IGM+1,IGM)
!
! -------------------------------------------------------------------------------------   
!     Do SVD of matrix H to determine if GMRES should stop
!
!           SVD destroys H, so make a copy
            Hcopy(1:IGM+1,1:IGM) = H(1:IGM+1,1:IGM) 
!
!           Call LAPACK SVD routine (this is a thin SVD)
            CALL DGESDD('A',IGM+1,IGM,Hcopy(1:IGM+1,1:IGM),IGM+1,       &
     &  Ssvd(1:IGM),Usvd(1:IGM+1,1:IGM+1),IGM+1,VHsvd(1:IGM,1:IGM),IGM, &
     &  WORKsvd,LWORKsvd,IWORKsvd,INFOsvd) 
!
! --------------------------------------------------------------------------------------
!       Invert H using SVD decompositon
!     
!           Set P = ||b|| U' e1
            P(1:IGM+1) = NORMb*Usvd(1,1:IGM+1)
!
!           Calculate X
            Xh(1:IGM) = P(1:IGM)/Ssvd(1:IGM)
!  
!  ---------------------------------------------------------------------------------------
!    Calculate GMRES residual
!
!    Set GMR = || H y - ||b|| e1 || / ||b|| = || Ssvd * Xh - P || / ||b||
            GMR = SQRT( SUM( (Ssvd(1:IGM)*Xh(1:IGM)-P(1:IGM))           &
     &               *(Ssvd(1:IGM)*Xh(1:IGM)-P(1:IGM)) )                &
     &          +  P(IGM+1)*P(IGM+1)  )/NORMb 
!
            print*,'GMRES step ',IGM
            print*,'| Ssvd * Xh - P | / |b| = ',GMR
!
            Yh(1:IGM) = matmul(                                         &
     &             TRANSPOSE( VHsvd(1:IGM,1:IGM) )  , Xh(1:IGM)   )
            Yh(1:IGM+1) = matmul( H(1:IGM+1,1:IGM),Yh(1:IGM) )
            Yh(1) = Yh(1) - NORMb
            GMR = SQRT( SUM( Yh(1:IGM+1)                                &
     &                            *Yh(1:IGM+1) ) )/NORMb 
!
            print*,' | H y - |b| e1 | / |b| = ',GMR
!
!$$$            Yh(1:IGM+1) = matmul(                                       &
!$$$     &              Usvd(1:IGM+1,1:IGM)  , Ssvd(1:IGM)*Xh(1:IGM)   )
!$$$            Yh(1) = Yh(1) - NORMb
!$$$            GMR = SQRT( SUM( Yh(1:IGM+1)                                & 
!$$$     &                            *Yh(1:IGM+1) ) )/NORMb 
!$$$!  
!$$$            print*,'GMRES convergence check:',                          &
!$$$     &               ' || U S x - ||b|| e1 || / ||b|| = ',GMR
!$$$!
!$$$            Yh(1:IGM) = matmul(                                         &
!$$$     &             TRANSPOSE( VHsvd(1:IGM,1:IGM) )  , Xh(1:IGM)   )
!$$$            Xh(1:IGM) = matmul(                                         &
!$$$     &              VHsvd(1:IGM,1:IGM)   , Yh(1:IGM)   )
!$$$            Yh(1:IGM+1) = matmul(                                       &
!$$$     &              Usvd(1:IGM+1,1:IGM)  , Ssvd(1:IGM)*Xh(1:IGM)   )
!$$$            Yh(1) = Yh(1) - NORMb
!$$$            GMR = SQRT( SUM( Yh(1:IGM+1)                                &
!$$$     &                            *Yh(1:IGM+1) ) )/NORMb 
!$$$!
!$$$            print*,'GMRES convergence check:',                          &
!$$$     &               ' || U S V^T y - ||b|| e1 || / ||b|| = ',GMR
!$$$!
!$$$            print*,'V*V^T = ',matmul(                                   &
!$$$     &         TRANSPOSE( VHsvd(1:IGM,1:IGM) ) , VHsvd(1:IGM,1:IGM) )
!
! ----------------------------------------------------------------------------------------
!
!    If residual is low enough exit GMRES loop
            IF (GMR.LE.GMTOL)  THEN
               GMREScount = GMREScount + IGM
               EXIT
            END IF
!
         ENDDO
!   ________________________________________________________________________________________
!   ----------------------------------------------------------------------------------------
!         GMRES Loop has now ended
!   ________________________________________________________________________________________
!   -----------------------------------------------------------------------------------------
!
!       *********************************
!        very last bit of arnoldi stuff
         IF (ArnoldiFLAG) EXIT
!      *********************************
! 
! -------------------------------------------------------------------------------------
!       Check if GMRES converged, if not exit
!           
         IF (IGM.EQ.GMRESMAX+1) THEN
           Print*,'GMRES did not converge, exiting Newton iteration'
           GMREScount = GMREScount + IGM
           WRITE(150,1502) IREST,'    ',                                 &
     &              TSTARTin,1._rk/v2in,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin, &
     & ' : "GMRES FAILED"  : ',iuposout+1,1._rk/v2N,TNEWT,SNEWTX,SNEWTY,SNEWTZ, &
     &                 NER,INE,GMREScount,HOOKcount  
            EXIT
         END IF
!
         Print*,'-----------------------------------------------------'
         Print*,'GMRES converged on GMRES step ',IGM
!
! ________________________________________________________________________
! ????????????????????????????????????????????????????????????????????????
! |||||||||    HOOK STEP          ||||||||||||||||||||||||||||||||||||||||
! ________________________________________________________________________
! 
!
         IHOOK = 0
         DO   ! this is a DO WHILE type of DO loop. The exit statements are at the end
!
            IHOOK = IHOOK + 1
            IF(IHOOK.GT.IHOOKMAX) THEN
               Print*,'HOOKSTEP failed to improve residual, '            
               Print*,'exiting Newton iteration'
               WRITE(150,1502) IREST,'    ',                             &
     &              TSTARTin,1._rk/v2in,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin, &
     & ' : "H-STEP FAILED" : ',iuposout+1,1._rk/v2N,TNEWT,SNEWTX,SNEWTY,SNEWTZ, &
     &                 NER,INE,GMREScount,HOOKcount    
               EXIT
            END IF
            Print*,'                   '
            Print*,'Performing new Hookstep ...'
! -------------------------------------------------------------------------------
!           FIND MU
!
!           Calculate X
            Xh(1:IGM) = P(1:IGM)/Ssvd(1:IGM)
!
!           First check ||x|| with mu = 0
            normX = SQRT( SUM( Xh(1:IGM)*Xh(1:IGM) ) )
            Print*,'Starting ||x|| = ',normX,', Delta = ',Delta
!          
!           Set MU = 0. for print statement below
            MUmid = 0._rk
!
            IF (normX.GT.root2*Delta) THEN 
!              Set intital value for MU1 = 0 and MU2 = maxval(Ssvd)
               MU1 = 0._rk
               MU2 = MAXVAL(Ssvd(1:IGM))
!
!              Recalculate ||x||
               Xh(1:IGM) = P(1:IGM)*Ssvd(1:IGM)                         &
     &              /( MU2 + Ssvd(1:IGM)*Ssvd(1:IGM) )
               normX = SQRT( SUM( Xh(1:IGM)*Xh(1:IGM) ) )
!
!              If ||x|| is still greater than delta then double MU and try again...
               DO 
               IF(normX.LT.Delta) EXIT
                  MU1 = MU2
                  MU2 = 2._rk*MU2
!                 Recalculate ||x||
                  Xh(1:IGM) = P(1:IGM)*Ssvd(1:IGM)                      &
     &                 /( MU2 + Ssvd(1:IGM)*Ssvd(1:IGM) )
                  normX = SQRT( SUM( Xh(1:IGM)*Xh(1:IGM) ) )
               ENDDO
!
!             normX has just been calculated with MU2 so set MUmid = MU2
               MUmid = MU2
!
!      Now MU1 should give ||x|| greater than delta and MU2 should give ||x|| less than delta
! 
!      Bisection root finding algorithm to find MU such that delta/root2 < ||x|| < root2*delta              
               DO
               IF(normX.LT.root2*Delta .AND. normX.GT.Delta/root2) EXIT
                  MUmid = 0.5_rk*(MU2+MU1)
!                 Recalculate ||x||
                  Xh(1:IGM) = P(1:IGM)*Ssvd(1:IGM)                      &
     &                 /( MUmid + Ssvd(1:IGM)*Ssvd(1:IGM) )
                  normX = SQRT( SUM( Xh(1:IGM)*Xh(1:IGM) ) )
                  IF (normX.GT.Delta) THEN
                     MU1 = MUmid
                  ELSE
                     MU2 = MUmid
                  ENDIF
               ENDDO
!
            END IF
!          
            print*,'Final ||X|| = ',normX,' Mu = ',MUmid
!            Print*,'H(1:IGM+1,1:IGM) = ',H(1:IGM+1,1:IGM)
!            Print*,'Ssvd(1:IGM) = ',Ssvd(1:IGM)
!            Print*,'Usvd(1:IGM+1,1:IGM+1) = ',Usvd(1:IGM+1,1:IGM+1) 
!            Print*,'VHsvd(1:IGM,1:IGM) = ',VHsvd(1:IGM,1:IGM)
!  ------------------------------------------------------------------------
!        NOW MU has been found check if Delta was appropriate.
!
!        Use VHsvd to calculate y = Vx
         Yh(1:IGM) = matmul(                                            &
     &          TRANSPOSE( VHsvd(1:IGM,1:IGM) ) , Xh(1:IGM)   )
!
         DIFFZ = CMPLX(0._rk,0._rk,rk)
         DIFFT = 0._rk
         DIFFSx = 0._rk
         IF(arclcFLAG2) DIFFv2 = 0._rk
         DO IORTH=1,IGM
            WHERE (L3) DIFFZ = DIFFZ +                                  &
     &                        q(:,IORTH) * CMPLX(Yh(IORTH),0._rk,rk)
            IF(.NOT.FixedP) DIFFT = DIFFT + qN1(IORTH) * Yh(IORTH) 
            DIFFSx = DIFFSx + qN2(IORTH) * Yh(IORTH) 
            IF(arclcFLAG2) DIFFv2 = DIFFv2 + qN3(IORTH) * Yh(IORTH)
         ENDDO
!
! --------------------------------------------------------------------------------
!      Calculate J*DIFF
!
         Print*,'Calculating J*DIFF ...'
!           Calculate the main part of JF*DIFF
            WHERE (L3) ZO =  ZNEWT + epsilon*DIFFZ
            IF(arclcFLAG2) THEN
               v2 = v2N + epsilon*DIFFv2
!        update NU                                                                                                    
               DO IKT = 1, NKT
                  IF(.NOT.LL(IKT))     CYCLE
                  WK = SQRT(KX(IKT)**2+KY(IKT)**2+KZ(IKT)**2)
                  R3 = log10(v2)+2._rk*log10(WK)
                  NU(IKT)     =  10._rk**R3
               ENDDO
            ENDIF
!
            TSTART = 0._rk
            DELT = TRUEDELT
            NSTOP  =   INT(TNEWT/DELT) !Note this rounds down

            DO IKT =1,NKT
               IF (LL(IKT))THEN
                  NUZN(IKT) = (1._rk - 0.5_rk*NU(IKT)*DELT) / DELT
                  NU1(IKT) = DELT / (1._rk + 0.5_rk*NU(IKT)*DELT)

                  XII(IKT) = ZO(IKT)
                  ETA(IKT) = ZO(NKT+IKT)
                  ZET(IKT) = ZO(2*NKT+IKT)
               ENDIF
            ENDDO

            CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);
!
!       Now do a shortened last time step to adjust period to TNEWT
            TSTART = NSTOP*DELT
            DELT = TNEWT - NSTOP*DELT
            IF (DELT.NE.0._rk) THEN
               WHERE (LL)
                  NUZN = (1._rk - 0.5_rk*NU*DELT) / DELT
                  NU1 = DELT / (1._rk + 0.5_rk*NU*DELT)
               END WHERE
               NSTOP = 1
            CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);
            END IF

!        Store final Z state
         DO IKT=1,NKT
            IF(LL(IKT))THEN
               ZN(IKT)       = XII(IKT)
               ZN(NKT+IKT)   = ETA(IKT)
               ZN(2*NKT+IKT) = ZET(IKT)
            ENDIF
         ENDDO

!     Calculate rows 1 to N of vector v in GMRES
            DO IKT = 1, NKT
               IF(.NOT.LL(IKT))     CYCLE
               SHIFT= EXP(-ZI*KX(IKT)*SNEWTX)*EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ)

               DO IT = 1,3
                  IK = (IT-1)*NKT +IKT
                  
                  JFDIFF(IK) = SHIFT*( ZN(IK)-ZNN(IK) )/epsilon - DIFFZ(IK) &
                       &      + dZNdSx(IK)*DIFFSx
                  IF(.NOT.FixedP) JFDIFF(IK) = JFDIFF(IK) + dZNdt(IK)*DIFFT
               ENDDO
            ENDDO
            !
!           Calculate v(N+1), v(N+2) and v(N+3)          
!            IF(.NOT.FixedP) FN1pred = SUM( REAL(CONJG(dzdt)*DIFFZ),L3)    
!            FN2pred = SUM( REAL(CONJG(dzdSx)*DIFFZ),L3 )
!            FN3pred = SUM( REAL(CONJG(dzdSz)*DIFFZ),L3 )
!
!     Approimate gradient of NORMFN with respect to z
!     If this is small it suggests a minimum in NORMFN has been found 
            Print*,'|| J*DIFF || / || DIFF || = ',                          &
     &          SUM( REAL(JFDIFF*CONJG(JFDIFF)), L3 ) /                      &
     &          SUM( REAL(DIFFZ*CONJG(DIFFZ)), L3 )
!
            WHERE (L3) FNpred = FN + JFDIFF
            NORMFNpred = SUM( REAL(FNpred*CONJG(FNpred)), L3 ) 
!            NORMFNpred = NORMFNpred + FN1pred**2 + FN2pred**2 + FN3pred**2
            NORMFNpred = SQRT(NORMFNpred)
            Print*,'Predicted || F{N+1} || = || FN + J*DIFF || =',      &
     &                                                        NORMFNpred
!
!        Check if timestepper has produced NAN
!$$$            IF (NORMFNpred.NE.NORMFNpred) EXIT
!            
            WHERE (L3) FNpredC = FN + const*JFDIFF
            NORMFNpredC = SUM( REAL(FNpredC*CONJG(FNpredC)), L3 ) 
!$$$            NORMFNpredC = NORMFNpredC +  (const*FN1pred)**2
            NORMFNpredC = SQRT(NORMFNpredC)
            Print*,'|| FN + const*J*DIFF || = ',NORMFNpredC
!
            Print*,'|| FN || = ',NORMFN 
!
            NORMbpredC = NORMFNpredC
!
            IF(arclcFLAG2) NORMbpredC =                                  &
     &                       SQRT(NORMFNpredC*NORMFNpredC + Narc*Narc)
!
! ------------------------------------------------------------------------------
!     Calculate UPO error with updated Z and T
!
!        Calculate potential new Z & T
         WHERE (L3) Zplus = ZNEWT + DIFFZ
         Tplus = TNEWT + DIFFT
         Sxplus = SNEWTX + DIFFSx
         IF(arclcFLAG2) v2plus = v2N + DIFFv2
!
!        Check if delta is appropriate
!
!        Set ZO = ZNEWT (ZO gets updated during timestepping)
         WHERE (L3) ZO = Zplus  
         IF(arclcFLAG2) THEN
            v2 = v2plus
!        update NU                                                                                                     
            DO IKT = 1, NKT
               IF(.NOT.LL(IKT))     CYCLE
               WK = SQRT(KX(IKT)**2+KY(IKT)**2+KZ(IKT)**2)
               R3 = log10(v2)+2._rk*log10(WK)
               NU(IKT)     =  10._rk**R3
            ENDDO
         ENDIF
!
         TSTART = 0._rk
         DELT = TRUEDELT
         NSTOP  =   INT(Tplus/DELT)  !Note this rounds down
         print*,'Calculating UPO'
         Print*,'Period = ',Tplus,', dt = ',DELT,', NSTOP = ',NSTOP 


            DO IKT =1,NKT
               IF (LL(IKT))THEN
                  NUZN(IKT) = (1._rk - 0.5_rk*NU(IKT)*DELT) / DELT
                  NU1(IKT) = DELT / (1._rk + 0.5_rk*NU(IKT)*DELT)

                  XII(IKT) = ZO(IKT)
                  ETA(IKT) = ZO(NKT+IKT)
                  ZET(IKT) = ZO(2*NKT+IKT)
               ENDIF
            ENDDO

            CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);
!
!       Now do a shortened last time step to adjust period to TNEWT
            TSTART = NSTOP*DELT
            DELT = TPlus - NSTOP*DELT
            IF (DELT.NE.0._rk) THEN
               WHERE (LL)
                  NUZN = (1._rk - 0.5_rk*NU*DELT) / DELT
                  NU1 = DELT / (1._rk + 0.5_rk*NU*DELT)
               END WHERE
               NSTOP = 1
               CALL TIMESTEP_CUDA(XII,ETA,ZET,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,KF,v2,IKF,IKN,L,NSTOP);
            END IF

!        Store final Z state
         DO IKT=1,NKT
            IF(LL(IKT))THEN
               ZN(IKT)       = XII(IKT)
               ZN(NKT+IKT)   = ETA(IKT)
               ZN(2*NKT+IKT) = ZET(IKT)
            ENDIF
         ENDDO
!
!        Shift final state back in x-direction
         DO IKT = 1, NKT
            IF(.NOT.LL(IKT))     CYCLE
            SHIFT = EXP(-ZI*KX(IKT)*Sxplus)*EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ)
            DO IT = 1,3
               IK = (IT-1)*NKT +IKT
               ZNSHIFT(IK) =SHIFT*ZN(IK)
            ENDDO
         ENDDO
!
!        Calculate F = Z(z,T) - z
         WHERE (L3) FNplus = ZNSHIFT - Zplus       
!
         NORMFNplus = SQRT( SUM( REAL(FNplus*CONJG(FNplus)), L3 ) )
         Print*,'|| FN{N+1} || = ',NORMFNplus
!
         NORMbplus = NORMFNplus
         IF (arclcFLAG2) THEN
            Narcplus = SUM( REAL(dZrdr*CONJG(Zplus-Zr2)), L3 )           &
                 &    + dSrdr*(Sxplus-Sr2) + dv2rdr*(v2plus-v2r2)                     &
                 &    - Ndelta_r
            IF(.NOT.FixedP) Narcplus = Narcplus + dTrdr*(Tplus-Tr2)
!                                                                                                                  
            NORMbplus = SQRT(NORMFNplus*NORMFNplus + Narcplus*Narcplus)
         END IF
         Print*,'|| b || = ',NORMb   
         Print*,'|| b{N+1} || = ',NORMbplus  
!
! ----------------------------------------------------------------------------------------------
!       Now check new solution to see if it is better than the old one
!
!        Check if timestepper has produced NAN
!$$$         IF (NORMFNplus.NE.NORMFNplus) EXIT
        IF (NORMFNplus.NE.NORMFNplus .OR. NORMFNpred.NE.NORMFNpred) THEN                 
           Print*,'Time-stepping routine produced NAN'                                      
           Print*,'Decrease trust region (DELTA = DELTA/2 )',                &              
     &                                             ' and try again ...'            
           DELTA = DELTA / 2._rk                                                            
           halveFLAG = .TRUE.                                                               
           CYCLE                                                                            
        END IF                  
!
!        Check if residual has increased ...
!$$$         IF (NORMFNplus .GE. NORMFNpredC) THEN
!$$$            Print*,'||FN{N+1}|| is .GE. || FN + const*J*DIFF ||'
        IF (NORMbplus .GE. NORMbpredC) THEN
            Print*,'||b{N+1}|| is .GE. || b ||'
!           Residual has increased
!
!           If a satisfactory solution has already been found then 
!           revert back to it and exit Hook step
            IF (doubleFLAG) THEN
               Print*,'Reverting back to previous saved solution'
               WHERE (L3) Zplus = Zplus2
               Tplus = Tplus2
               Sxplus = Sxplus2
               IF(arclcFLAG2) v2plus = v2plus2
               WHERE (L3) ZN = ZN2
               WHERE (L3) FNplus = FNplus2
               NORMFNplus = NORMFNplus2
               DELTA = DELTA / 2._rk
               EXIT
            END IF
!               
!           If not decrease trust region 
            Print*,'Decrease trust region (DELTA = DELTA/2 )',                &
     &                                              ' and try again ...'
            DELTA = DELTA / 2._rk
            halveFLAG = .TRUE.
         ELSE 
!$$$            Print*,'||F|| is .LT. || FN + const*J*DIFF ||'
            Print*,'||b{N+1}|| is .LT. || b ||'
!           Residual has decreased
!
            IF (halveFLAG) THEN
               print*,'Accept current solution'
               EXIT
            END IF
!           
            IF (doubleFLAG .AND. NORMFNplus.GT.NORMFNplus2) THEN 
               Print*,'New ||F|| is greater than previous ||F||'
               Print*,'Reverting back to previous saved solution'
               WHERE (L3) Zplus = Zplus2
               Tplus = Tplus2
               Sxplus = Sxplus2
               WHERE (L3) ZN = ZN2
               WHERE (L3) FNplus = FNplus2
               NORMFNplus = NORMFNplus2
               DELTA = DELTA / 2._rk
               EXIT
            END IF
!            
            Print*,'Considering whether to increase trust region ...'
            IF (MUmid .EQ. 0._rk) THEN
               Print*,'Mu = 0 => ',                                      &
     &              'Trust region is already larger than step size'
               Print*,'Accept current solution'
               EXIT
            ELSE IF ( ABS(NORMFNpred-NORMFNplus) .LT.                   &
     &                         0.1_rk*ABS(NORMFNplus-NORMFN) ) THEN
               Print*,'Prediction is too good, increasing trust region'
               DELTA = DELTA * 2._rk
               doubleFLAG = .TRUE.
               Print*,'Saving current solution before retrying'
               WHERE (L3) Zplus2 = Zplus
               Tplus2 = Tplus
               Sxplus2 = Sxplus               
               IF(arclcFLAG2) v2plus2 = v2plus
               WHERE (L3) ZN2 = ZN
               WHERE (L3) FNplus2 = FNplus
               NORMFNplus2 = NORMFNplus
            ELSE IF (NORMFNPLUS .LT. NORMFNpred) THEN
               Print*,'Negative curvature, increasing trust region'
               DELTA = DELTA * 2._rk
               doubleFLAG = .TRUE.
               Print*,'Saving current solution before retrying'
               WHERE (L3) Zplus2 = Zplus
               Tplus2 = Tplus
               Sxplus2 = Sxplus
               IF(arclcFLAG2) v2plus2 = v2plus
               WHERE (L3) ZN2 = ZN
               WHERE (L3) FNplus2 = FNplus
               NORMFNplus2 = NORMFNplus
            ELSE
               Print*,'Accept current solution'
               EXIT
            END IF
         ENDIF
      ENDDO
      HOOKcount = HOOKcount + IHOOK
!
!! ________________________________________________________________________
! ????????????????????????????????????????????????????????????????????????
! |||||||||    HOOK STEP FINISHED   ||||||||||||||||||||||||||||||||||||||
! ________________________________________________________________________
!
!
      halveFLAG = .FALSE.
      doubleFLAG = .FALSE.
!
!     Check if Timestepping has blown up
      IF (NORMFNpred.NE.NORMFNpred .OR. NORMFNplus.NE.NORMFNplus) THEN 
         Print*,'Time-stepping routine produced NAN'
         IF (TRUEDELT .LT. 0.1_rk*TrueTRUEDELT) THEN
            Print*,'Exiting Newton iteration'
            WRITE(150,1502) IREST,'    ',                                &
     &              TSTARTin,1._rk/v2in,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin, &
     & ' : "T-STEP FAILED" : ',iuposout+1,1._rk/v2N,TNEWT,SNEWTX,         &
     &                 SNEWTY,SNEWTZ,NER,INE,GMREScount,HOOKcount 
            EXIT
         ENDIF
         Print*,'Reducing dt and retrying newton step.'
         TRUEDELT = TRUEDELT/2._rk
         changeDELT = .TRUE.
!        Shift final state back in x-direction
         DO IKT = 1, NKT
            IF(.NOT.LL(IKT))     CYCLE
            SHIFT = EXP(-ZI*KX(IKT)*SNEWTX)*EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ)
            DO IT = 1,3
               IK = (IT-1)*NKT +IKT
               ZNSHIFT(IK) = SHIFT*ZNN(IK)
            ENDDO
         ENDDO
         CYCLE
      ENDIF 
!
      IF(IHOOK.GT.IHOOKMAX) EXIT   
!
! --------------------------------------------------------------------------------
!
!        Store final Z state 
         WHERE (L3) ZNN = ZN
!
!        Update Z and T for next newton step 
         WHERE (L3) ZNEWT = Zplus
         TNEWT = Tplus
         SNEWTX = Sxplus
         IF(arclcFLAG2) v2N   = v2plus
!
!        Shift final state back in x-direction
        DO IKT = 1, NKT
            IF(.NOT.LL(IKT))     CYCLE
            SHIFT = EXP(-ZI*KX(IKT)*SNEWTX)*EXP(-ZI*KY(IKT)*SNEWTY)*EXP(-ZI*KZ(IKT)*SNEWTZ)
            DO IT = 1,3
               IK = (IT-1)*NKT +IKT
               ZNSHIFT(IK) = SHIFT*ZNN(IK)
            ENDDO
         ENDDO
!     
         NORMZ = SQRT( SUM( REAL(ZNN*CONJG(ZNN)), L3 ) )
!
!      Calculate 1st residual
         WHERE (L3) FN = FNplus
         NORMFN = NORMFNplus
         NORMb = NORMFN
         Print*,'||FN|| = ',NORMFN
!
         IF(arclcFLAG2) THEN
!        Calculate new Narc                                                                                                          
            Narc = SUM( REAL(dZrdr*CONJG(ZNEWT-Zr2)), L3 )              &
     &   + dSrdr*(SNEWTX-Sr2) + dv2rdr*(v2N-v2r2)                        &
     &   - Ndelta_r
            IF(.NOT.FixedP) Narc = Narc + dTrdr*(TNEWT-Tr2)
            Print*,'Narc = ',Narc
            NORMb = SQRT(NORMFN*NORMFN + Narc*Narc)
         END IF
         Print*,'||b|| = ',NORMb
!
!
!        Use NormFN to check for convergence 
         NER = NORMFN/NORMZ
         Print*,'  '
         Print*,'-----------------------------------------------------'
         PRINT*,'Newton convergence check: ||F|| / ||Z|| = ',NER
         Print*,'-----------------------------------------------------'
         Print*,'  '
         Print*,'||Z(T)|| = ',SQRT( SUM( REAL(ZNN*CONJG(ZNN)), L3 ) )
         Print*,'||Z(0)|| = ',SQRT( SUM( REAL(Zplus*CONJG(Zplus)), L3 ))
         Print*,'  '
!
!
!         write(99) 0._rk,v2N,TNEWT,SNEWTX,SNEWTY,SNEWTZ,NER,                          &
!     &                         (ZNEWT(IKT),IKT=1,3*NKT)
!         izkout = izkout +1
!         print*,'Wrote to Zk.out number ',izkout
         print*,'period =',TNEWT,'X shift =',SNEWTX,'Z shift =',SNEWTZ
         IF(arclcFLAG) Print*,'Re = ',1._rk / v2N
         Print*,'-----------------------------------------------------'
         
         NERold = abs((NER-NERold)/NER)
         if(NERold .lt. 0.001_rk)then
            NERcount = NERcount+1

            if(NERcount .gt. 5)then
               WRITE(150,1502) IREST,'    ',                                &
     &             TSTARTin,1._rk/v2in,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin, &
     & ' : "NEWTON STALLED" : ',iuposout+1,1._rk/v2N,TNEWT,SNEWTX,SNEWTY,SNEWTZ, &
     &                 NER,INE,GMREScount,HOOKcount
               EXIT
            END IF
         ELSE
            NERcount = 0
         END IF
         
         NERold = NER
!
!        If Newton method has converged then reset everything and exit
         IF (NER.LE.NETOL)  THEN
            Print*,'-------------------------------------------------'
            Print*,'-------------------------------------------------'
            Print*,'  '
            PRINT*,'Newton method converged in',INE,' steps.'         
            WRITE(150,1502) IREST,'    ',                                &
     &              TSTARTin,1._rk/v2in,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin, &
     & ' :   "CONVERGED"   : ',iuposout+1,1._rk/v2N,TNEWT,SNEWTX,         &
     &                 SNEWTY,SNEWTZ,NER,INE,GMREScount,HOOKcount
            convergeFLAG = 1
            IF(arclcFLAG) THEN
               CONVERGED = .TRUE.
               IF(SecondRun) SecondRun = .FALSE.
               IF(FirstRun) THEN
                  FirstRun = .FALSE.
                  SecondRun = .TRUE.
                  arclcFLAG2 = .TRUE.
               END IF
               WHERE (L3) Zr0 = Zr1
               Tr0 = Tr1
               Sr0 = Sr1
               v2r0 = v2r1
               WHERE (L3) Zr1 = Zr2
               Tr1 = Tr2
               Sr1 = Sr2
               v2r1 = v2r2
               delta_r1 = delta_r2
               WHERE (L3) Zr2 = ZNEWT
               Tr2 = TNEWT
               Sr2 = SNEWTX
               v2r2 = v2N
               delta_r2 = Ndelta_r
            END IF
            EXIT
         END IF
!
!        If max newton steps have been reached then admit failure and move on to next guess
         IF (INE.EQ.NEWTONMAX)  THEN
            WRITE(150,1502) IREST,'    ',                                &
     &             TSTARTin,1._rk/v2in,TNEWTin,SNEWTXin,SNEWTYin,SNEWTZin,NERin, &
     & ' : "NEWTON FAILED" : ',iuposout+1,1._rk/v2N,TNEWT,SNEWTX,SNEWTY,SNEWTZ, &
     &                 NER,INE,GMREScount,HOOKcount 
         END IF
!
      ENDDO
!
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! ----------------------------------------------------------------------------
!     NEWTON ITERATIONS END
! ----------------------------------------------------------------------------    
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!
!  
!     Write out final state of UPO to file
      if( convergeFLAG .eq. 1) then 
         write(999) 0._rk,v2N,TNEWT,SNEWTX,SNEWTY,SNEWTZ,NER,                            & 
              &     (ZNEWT(IKT),IKT=1,3*NKT)
         iuposout = iuposout + 1
         print*,'Wrote to UPOs.out number ',iuposout
         print*,'period =',TNEWT,'X shift =',SNEWTX,'Z shift =',SNEWTZ
         IF(arclcFLAG) Print*,'Re = ',1._rk / v2N
         Print*,'-----------------------------------------------------'
         convergeFLAG = 0
      endif
!________
!
! WRAP UP
!________
!
!     OUTPUT FINAL STATE 
!
!
!
      CALL CPU_TIME(toc)
      time3 = toc-tic
      print*,'    '
      write(6,5000) time3, time3/60._rk, time3/3600._rk
 5000  format(1x,'CPU time required for main loop = ',F10.3,' s = ',      &
      &               F7.3,' m = ',F7.3,' h.')
      print*,'    '
!      print*,'FFTW FFT time  ',MYTIME,' s'
!
!
      CALL FLUSH(150)
!
!      close(99)
!
      close(13)
      close(31)
!
      ENDDO
!
      close(150)
      close(999)
!
      IF (changeDELT) THEN
         print*,'WARNING: dt was changed for 1 or more NEWTON attempts.'
      END IF
!
      print*,'Finished'
!
 1501 FORMAT(A,A,A,A,A,A,A,A,A,A,A,A)  
 1502 FORMAT(I5,A,E16.6,F10.5,F12.5,F12.5,F12.5,F12.5,E14.5,                        & 
     &                      A,I4,F12.5,F10.5,F10.5,F10.5,F10.5,E14.5,I6,I6,I6)     
!
      END PROGRAM MAIN
!
