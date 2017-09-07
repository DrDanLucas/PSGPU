PROGRAM MAIN 
!
      use GLOBAL_PARAMS
      use PARAMETERS
      IMPLICIT NONE 
!
      COMPLEX(rk), DIMENSION(IKTX,IKTY) :: ZN,ZO,ZNEWT
      COMPLEX(rk), DIMENSION(IKTX,IKTY) :: ZNSHIFT,FN
!
      REAL(rk), DIMENSION(NX,NY)     :: ZR
      REAL(rk), DIMENSION(IKTX,IKTY)     :: NUZN,NU1,KX,KY
      REAL(rk) :: RANNO,TSTART,TIME
      REAL(rk) :: KKX,KKY,WK
      REAL(rk) :: NER,NORMFN,NORMZ
      REAL(rk) :: TNEWT,PERIODin,SHIFTXin,SHIFTYin,NERin,NER2in
!
      INTEGER, PARAMETER ::  NX2=NX/2+1
      INTEGER ::  IKF(NX2,NY), IKN(NX2*NY)
      INTEGER ::  IKX,IKY,jw
      INTEGER ::  NT,IT1,IT2,ITR,INKY,ITC,IKK,IK,IT,NIT
      INTEGER ::  i,j,ix,jx,m,ic,jc,biggest,id,jd,nvort

!
      print*,'rk = ',rk
      print*,'Truncation wavenumber = ',(REAL(kty,rk)-0.5_rk)
!
! -------------------------------------------------------------------
!
!     Read in input parameters from file and set constants.
      CALL params_read_txt()
!
! -----------------------------------------------------------------
!
!
! INITIALISE STUFF, GET THE FFT READY, ETC.
!
      TIME  = RANNO(ISEED) !Initializes random number generator 
!
      IF (NOUT.EQ.0) NOUT = 1
      IF (NOUTV.EQ.0) NOUTV = 1
!
!
      IF ( AMPV.NE.0._rk                                 &
     &                .AND.IREST.EQ.0) THEN
        jw = nstop/nout + 1
      ELSE
        jw = nstop/nout
      ENDIF
!
! ------------------------------------------------------------------------------
!     Initialize all output files
      print*,'   '
      print*,'Files will contain ',jw,' output times.'
      print*,'   '
      open (99,file='Zk.out',form='unformatted')
      write(99) 1,IKTX,IKTY
!     Initialize variables
      NT = 0
      ZN  = CMPLX(0._rk,0._rk,rk)
      ZO  = CMPLX(0._rk,0._rk,rk)

!
      PERIODin = 0._rk
      SHIFTXin = 0._rk
      SHIFTYin = 0._rk
      NERin = 0._rk
      NER2in = 0._rk
      if (irest.eq.0) TSTART=0._rk
!
!     Initialize arrays and if irest.NE.0 load starting state from file
      CALL INIT (ZO,TSTART,PERIODin,SHIFTXin,SHIFTYin,NERin)    
!
      IF (UPOflag.eq.1) THEN
         print*,'        Period=',PERIODin
         print*,'        SHIFTX=',SHIFTXin
         print*,'        SHIFTY=',SHIFTYin
         print*,'   Wave Speed =',SHIFTXin/PERIODin
         print*,'      Residual=',NERin
         print*,'               '
         TSTEP = PERIODin
         TSTART = 0._rk
      ENDIF

!     Output to Screen
      print*,'        alpha=',alpha
      print*,'           Re=',1./v2
      print*,'          ktx=',ktx
      print*,'          kty=',kty
      print*,'          NSTOP=',NSTOP
      print*,'          NOUT=',NOUT
      print*,'          NOUTV=',NOUTV
!
! -------------------------------------------------------------------------
!
!
      print*,'Number of degrees of freedom = ',2*COUNT(LL)
      print*,'Max coefficient of initial condition = ',                 &
     &                                               MAXVAL(ABS(ZO),LL)
!
!

! ---------------------------------------------------------------------------
!     OUTPUT INITIAL STATE 
! --------------------------------------------------------------------------
!
      TIME = TSTART
      write(99) TIME,v2,0._rk,0._rk,0._rk,0._rk,                           & 
     &                             ((ZO(IKX,IKY),IKX=1,IKTX),IKY=1,IKTY)
      izkout = 1
      print*,'Wrote to Zk.out number ',izkout,' at time ',TIME
!
      CALL FLUSH(6)
!
! --------------------------------------------------------------------------
! ------------     TIME STEPPING  ------------------------------------------
! --------------------------------------------------------------------------
!
!     Store initial state, because ZO gets updated during timestepping
      WHERE(LL) ZNEWT = ZO
!
      NSTOP  =   INT(TSTEP/DELT) !Note this rounds down
      Print*,'Start time = ',TSTART
      Print*,'Time increase = ',TSTEP,', Time step = ',DELT
      Print*,'Number regular time steps = ',NSTOP 
      Print*,'Number of output steps  = ',nout 
!
!      CALL etime(time2,tic)
      CALL CPU_TIME(tic)
      CALL SYSTEM_CLOCK(IT1,ITR)

      call set_gpu(0)

!     Here we set arrays to allow kernels in cuda to be non-problem specific
!     i.e. these arrays give all the required info for a specific element ik in linear memory.
!     and do not duplicate the work. Also doing this on the C side of the timestep_cuda routine
!     resulted in a seg fault for large problems, probably due to memory allocation with the 
!     fortran/C compilation, in short this was found to be the most robust method!
!     Also should go into init or a separate subroutine at some point....

      DO IKX=1,IKTX
         KKX = REAL(IKX - 1,rk)*alpha
         DO IKY=1,IKTY
            KKY = REAL(IKY - KTY - 1,rk)
            IK = (IKY-1)*IKTX+IKX

!           store wavenumbers
            KX(IKX,IKY)=KKX
            KY(IKX,IKY)=KKY
            WK = KKX*KKX + KKY*KKY

!           store timestepping arrays
            IF(LL(IKX,IKY))THEN
               NUZN(IKX,IKY)= (1._rk - 0.5_rk*v2*WK*DELT) / DELT
               NU1(IKX,IKY) =DELT / (1._rk + 0.5_rk*v2*WK*DELT) 
            ELSE
               NUZN(IKX,IKY)=0._rk
               NU1(IKX,IKY)=0._rk
            ENDIF
         ENDDO
      ENDDO


!     Store index array for padding before KR FFT
      DO IKX=1,IKTX
         DO IKY=1,KTY+1
            IKF(IKX,IKY) = IKX-1+IKTX*(IKY-1+KTY)
         ENDDO
         DO IKY=KTY+2,NY-KTY
            IKF(IKX,IKY) = -1
         ENDDO
         DO IKY=NY-KTY+1,NY
            IKF(IKX,IKY) = IKX-1+IKTX*(IKY-1+KTY-NY)
         ENDDO
      ENDDO
!
      DO IKY=1,NY
         DO IKX=IKTX+1,NX/2+1
            IKF(IKX,IKY) = -1
         ENDDO
      ENDDO

!     Store index array for rewriting and normalising after RK FFT
       DO IKX =1,IKTX
          DO IKY =1,KTY
             IK  = (IKY-1)*IKTX +IKX
             IKK = (NY+IKY-1-KTY)*NX2 + IKX-1

             IKN(IK) = IKK
         ENDDO
         DO IKY =KTY+1,IKTY
             IK  = (IKY-1)*IKTX +IKX
            IKK = (IKY-1-KTY)*NX2 +IKX-1

            IKN(IK) = IKK
         ENDDO
      ENDDO

!      open(12,file='vort1.dat',access="STREAM",form='UNFORMATTED')

!      do nt=1,140
!          read(12) time,((ZR(ikx,iky),ikx=1,NX),iky=1,NY)
!      enddo
!      print*,'time = ',time


      NIT = NSTOP
      CALL TIMESTEP_CUDA(ZO,ZR,NUZN,NU1,KX,KY,TIME,TSTART,               &
     &       AMPFOR,DELT,alpha,v2,ResidualThreshold,IKF,IKN,             &
     &       IKTX,IKTY,KTY,NX,NY,NIT,NOUT,NOUTV,L,RCFLAG,STFLAG)

      write(*,*) 'timestepping done'
      TSTART = TIME
!      ZN=ZO
!
!     Now do a shortened last time step to adjust to TSTEP exactly 
      TSTART = NSTOP*DELT
      DELT = TSTEP - NSTOP*DELT

      IF (DELT.NE.0._rk) THEN
         DO IKX=1,IKTX
            KKX = REAL(IKX - 1,rk)*alpha
            DO IKY=1,IKTY
               KKY = REAL(IKY - KTY - 1,rk)
               IK = (IKY-1)*IKTX+IKX
               WK = KKX*KKX + KKY*KKY
               !           store timestepping arrays
               IF(LL(IKX,IKY))THEN
                  NUZN(IKX,IKY)= (1._rk - 0.5_rk*v2*WK*DELT) / DELT
                  NU1(IKX,IKY) =DELT / (1._rk + 0.5_rk*v2*WK*DELT) 
               ELSE
                  NUZN(IKX,IKY)=0._rk
                  NU1(IKX,IKY)=0._rk
               ENDIF
            ENDDO
         ENDDO
         
         print*,'Calculating last time step of length: ',DELT
         write(*,*) 'TSTART= ',TSTART,' at time = ',TIME
         NSTOP = 1
         
      CALL TIMESTEP_CUDA(ZO,ZR,NUZN,NU1,KX,KY,TIME,TSTART,               &
     &       AMPFOR,DELT,alpha,v2,ResidualThreshold,IKF,IKN,             &
     &       IKTX,IKTY,KTY,NX,NY,NSTOP,NOUT,L,0,0)
      write(*,*) 'written number at time = ',TIME
      END IF
      ZN=ZO
! ----------------------------------------------------------------------------
!     Timing stuff
!      CALL etime(time2,toc)
      CALL CPU_TIME(toc)
      CALL SYSTEM_CLOCK(IT2,ITR)
      time3 = toc-tic
      print*,'    '
      write(6,5000) time3, time3/60._rk, time3/3600._rk
 5000  format(1x,'CPU time required for main loop = ',F7.3,' s = ',      &
     &               F7.3,' m = ',F7.3,' h.')
      print*,'    '
      print*, 'SYSTEM CLOCK TIME = ', (IT2-IT1)/ITR
!
! -------------------------------------------------------------------------
      write(99) TIME,v2,0._rk,0._rk,0._rk,0._rk,                           &
     &                             ((ZO(IKX,IKY),IKX=1,IKTX),IKY=1,IKTY)
      izkout = izkout +1
      print*,'Wrote to Zk.out number ',izkout,' at time ',TIME
      
! --------------------------------------------------------------------------
!     Calculate Residual for Guess or Converged UPO
! --------------------------------------------------------------------------
!
      IF (UPOflag.eq.1) THEN

!     Shift final state back in x-direction
         DO IKY = 1, IKTY
            KKY =  REAL(IKY - KTY - 1,rk)
            DO IKX = 1, IKTX
               IF(.NOT.LL(IKX,IKY))     CYCLE
               KKX =  REAL(IKX-1,rk)*alpha
               ZNSHIFT(IKX,IKY) = EXP(-ZI*KKX*SHIFTXin)*EXP(-ZI*KKY*SHIFTYin) &
                    &                                *ZO(IKX,IKY)
            ENDDO
         ENDDO
         !
         NORMZ = SQRT( SUM( REAL(ZO*CONJG(ZO)), LL ) )
         !
         !     Calculate 1st residual
         !     Calculate F = Z(z,T) - z
         WHERE (LL) FN = ZNSHIFT - ZNEWT
         !
         NORMFN = SQRT( SUM( REAL(FN*CONJG(FN)), LL ) )
         !
         !     Use NormFN to check for convergence
         NER = NORMFN/NORMZ
         !
         PRINT*,'Newton convergence check: ||F|| / ||Z|| = ',NER
         Print*,'-----------------------------------------------------'
         Print*, 'NORMZ  = ', NORMZ
         Print*, 'NORMFN = ', NORMFN
         Print*,'-----------------------------------------------------'
         !
         write(99) TIME,v2,0._rk,0._rk,0._rk,0._rk,                           &
              &                             ((ZNSHIFT(IKX,IKY),IKX=1,IKTX),IKY=1,IKTY)
         izkout = izkout +1
         print*,'Wrote to Zk.out number ',izkout,' at time ',TIME
         !        Check if calculated residuals match the ones from timestepping
         !        These should be an exact match, if everything is correct!
         IF (NER.ne.NERin) THEN
            print*,'WARNING: calculated ||F||/||Z|| does not match GMRES'
         ENDIF
      ENDIF

!
! -------------------------------------------------------------------------
! -------------------------------------------------------------------------
! -------------------------------------------------------------------------
      close(99)
!
      print*,'Finished'
!
 1501 FORMAT(A,A,A,A,A,A)  
 140  FORMAT(I6)  
! 
      END PROGRAM MAIN
!
