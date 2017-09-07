! -------------------------------------------------------------------                                                                                                                                                     
      PROGRAM MAIN
! -------------------------------------------------------------------                                                                                                                                                     
        ! Main program for CUDA pseudospectral DNS code.
        ! This verion is for the 3D Navier Stokes eqs
        ! Note FORTRAN front end is legacy to connect into
        ! the Newton-GMRES-hookstep code.                

        ! See README for details on usage.               
        !                                               
        ! Dan Lucas dl549@cam.ac.uk 2016                
! -------------------------------------------------------------------     
      use GLOBAL_PARAMS
      use PARAMETERS
      IMPLICIT NONE 
      include 'mpif.h'
!
      COMPLEX(rk), DIMENSION(NKT) :: XII,ETA,ZET
      REAL(rk), DIMENSION(NR)     :: XIR,ETR,ZTR
      REAL(rk), DIMENSION(NKT) :: KX,KY,KZ,NUZN,NU1,NU

      COMPLEX(rk):: SHIFT
      REAL(rk) :: RANNO,TSTART,TIME,FK
      REAL(rk) :: SHIFTXin,SHIFTYin,SHIFTZin,PERIODin
      REAL(rk) :: NORMZ,NORMFN,NER
!
      INTEGER ::  IKF(NX2*NY*NZ), IKN(NKT)
      INTEGER ::  IKX,IKY,IKZ,jw,KFX,KFY,KFZ,IYY,IZZ,IKT
      INTEGER ::  NT,IT1,IT2,ITR,INKY,IK,IKK
      INTEGER ::  i,j,ix,jx,m,ic,jc,biggest,id,jd,ISTART
! -------------------------------------------------------------------
      !This version of the code can run 2 jobs simultaneously on nodes
      !which have 2 GPUs to make efficient use of CPU quota           
      !This is handled with a pretty straightforward MPI implementation
      !each process is the host for each GPU and CUDA instance         
! ------------------------------------------------------------------- !
      CALL MPI_INIT(IERROR)
      CALL MPI_COMM_SIZE(MPI_COMM_WORLD,NPROCS,IERROR)
      CALL MPI_COMM_RANK(MPI_COMM_WORLD,RANK,IERROR)
      write(simdir,"(A3 I1)") "JOB",RANK
!
!     Read in input parameters from file and set constants.
      CALL params_read_txt()
! -----------------------------------------------------------------
!
      TIME  = RANNO(ISEED) !Initializes random number generator 
!
      IF (NOUT.EQ.0) NOUT = 1
      IF (NOUTV.EQ.0) NOUTV = 1 
      FK = KF
! ------------------------------------------------------------------------------
!     Initialize all output files
      open (99,file=trim(simdir)//'/Zk.out',form='unformatted')
      write(99) 1,IKTX,IKTY,IKTZ
      ISTART=IREST
      DO IREST = ISTART,ISTOP
!     Initialize variables
      NT = 0
      MYTIME =   0._4
      XII  = CMPLX(0._rk,0._rk,rk)
      ETA  = CMPLX(0._rk,0._rk,rk)
      ZET  = CMPLX(0._rk,0._rk,rk)
      TSTART=0._rk
!
!     Initialize arrays and if irest.NE.0 load starting state from file
      CALL INIT (XII,ETA,ZET,KX,KY,KZ,NU,NUZN,NU1,TSTART,PERIODin,SHIFTXin,SHIFTYin,SHIFTZin)

      IF (UPOflag.eq.1) THEN
         print*,'        Period=',PERIODin
         print*,'        SHIFTX=',SHIFTXin
         print*,'        SHIFTY=',SHIFTYin
         print*,'        SHIFTZ=',SHIFTZin
         print*,'               '
         TSTEP = PERIODin
         TSTART = 0._rk
      ENDIF
!
!     Output to Screen
      print*,'        alpha=',alpha
      print*,'           Re=',1./v2
      print*,'          ktx=',ktx
      print*,'          kty=',kty
      print*,'          ktz=',ktz
      print*,'          NSTOP=',NSTOP
      print*,'          NOUT=',NOUT
      print*,'          NOUTV=',NOUTV
!
! ---------------------------------------------------------------------------
!
      NSTOP  =   INT(TSTEP/DELT) !Note this rounds down
      Print*,'Start time = ',TSTART
      Print*,'Time increase = ',TSTEP,', Time step = ',DELT
      Print*,'Number regular time steps = ',NSTOP 
!
!      CALL etime(time2,tic)
      CALL CPU_TIME(tic)
      CALL SYSTEM_CLOCK(IT1,ITR)


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

!
      TIME = 0._rk
      CALL TIMESTEP_CUDA(XII,ETA,ZET,XIR,ETR,ZTR,NUZN,NU1,KX,KY,KZ,TIME,TSTART,AMPFOR,DELT,ResidualThreshold,&
     &                   FK,v2,alpha,IKF,IKN,L,IKTX,IKTY,IKTZ,KTZ,NKT,NX,NY,NZ,NSTOP,NOUT,NOUTV,STATSFLAG,RCFLAG,RANK)


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
   ENDDO
!
! ---------------------------------------------------------------------------

!________
!
! WRAP UP
!________
!
!
      close(99)
      print*,'Finished'      
      CALL MPI_FINALIZE(ierror)
      END PROGRAM MAIN
!
