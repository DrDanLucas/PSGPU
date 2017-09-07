!   _______________________________________________________
!
      SUBROUTINE KR_FFTW(ZK,XIR,ETR,ZTR,FF2,IKF,PlanK2XR,PlanK2YR,PlanK2ZR)
!
!      CALLS SPECTRAL -> GRID POINT TRANSFORMS.
!
      USE GLOBAL_PARAMS
      IMPLICIT NONE
!
      INTEGER         :: IKX,IKY,INKY,IKZ,INKZ,IKT
      INTEGER         :: IKF(NX2*NY*NZ)
      INTEGER(KIND=8) :: PlanK2XR,PlanK2YR,PlanK2ZR
!     
      REAL(rk)     :: XIR(NR),ETR(NR),ZTR(NR)
      COMPLEX(rk)  :: ZK(3*NKT),FF2(NX2*NY*NZ)
!
!    FIRST OF ALL MAKE SURE CONJUGATE SYMMETRY IS IMPOSED
!
      DO IKY=1,KTY
         INKY = - IKY + 2*KTY + 1
!      On Kx=0,Kz=0, set -Ky values to conjugate of +Ky values
         ZK((IKY-1)*IKTX+1) = CONJG(ZK(INKY*IKTX+1))
         ZK(NKT+(IKY-1)*IKTX+1) = CONJG(ZK(NKT+INKY*IKTX+1))
         ZK(2*NKT+(IKY-1)*IKTX+1) = CONJG(ZK(2*NKT+INKY*IKTX+1))
         DO IKZ=1,KTZ
            INKZ = - IKZ + 2*KTZ + 1
!      On Kx=0,set -Ky,-Kz values to conjugate of +Ky,+Kz values
            ZK(((IKZ-1)*IKTY+IKY-1)*IKTX+1) = CONJG(ZK((INKZ*IKTY+INKY)*IKTX+1))
            ZK(NKT+((IKZ-1)*IKTY+IKY-1)*IKTX+1) = CONJG(ZK(NKT+(INKZ*IKTY+INKY)*IKTX+1))
            ZK(2*NKT+((IKZ-1)*IKTY+IKY-1)*IKTX+1) = CONJG(ZK(2*NKT+(INKZ*IKTY+INKY)*IKTX+1))
!      On Kx=0,set -Ky,+Kz values to conjugate of +Ky,-Kz values
            ZK((INKZ*IKTY+IKY-1)*IKTX+1) = CONJG(ZK(((IKZ-1)*IKTY+INKY)*IKTX+1))
            ZK(NKT+(INKZ*IKTY+IKY-1)*IKTX+1) = CONJG(ZK(NKT+((IKZ-1)*IKTY+INKY)*IKTX+1))
            ZK(2*NKT+(INKZ*IKTY+IKY-1)*IKTX+1) = CONJG(ZK(2*NKT+((IKZ-1)*IKTY+INKY)*IKTX+1))
         ENDDO
      ENDDO
      DO IKZ=1,KTZ
         INKZ = - IKZ + 2*KTZ + 1
         !      On Kx=0,Ky=0 set -Kz values to conjugate of +Kz values
         ZK((IKZ-1)*IKTY*IKTX+1) = CONJG(ZK(INKZ*IKTY*IKTX+1))
         ZK(NKT+(IKZ-1)*IKTY*IKTX+1) = CONJG(ZK(NKT+INKZ*IKTY*IKTX+1))
         ZK(2*NKT+(IKZ-1)*IKTY*IKTX+1) = CONJG(ZK(2*NKT+INKZ*IKTY*IKTX+1))
      ENDDO
!
!      Map -Ky values to -Ky+NY and fill trucated wave numbers with zeros
      DO IKT =1,NX2*NY*NZ
         IF(IKF(IKT) .eq. -1)THEN
            FF2(IKT)=CMPLX(0._rk,0._rk,rk)
         ELSE
            FF2(IKT)= ZK(IKF(IKT)+1)
         ENDIF
      ENDDO
! _______________________________________________________________

!     EXECUTE THE FFT
      call dfftw_execute(PlanK2XR)
! _______________________________________________________________
!      Map -Ky values to -Ky+NY and fill trucated wave numbers with zeros
      DO IKT =1,NX2*NY*NZ
         IF(IKF(IKT) .eq. -1)THEN
            FF2(IKT)=CMPLX(0._rk,0._rk,rk)
         ELSE
            FF2(IKT)= ZK(NKT+IKF(IKT)+1)
         ENDIF
      ENDDO

      call dfftw_execute(PlanK2YR)
!      Map -Ky values to -Ky+NY and fill trucated wave numbers with zeros
      DO IKT =1,NX2*NY*NZ
         IF(IKF(IKT) .eq. -1)THEN
            FF2(IKT)=CMPLX(0._rk,0._rk,rk)
         ELSE
            FF2(IKT)= ZK(2*NKT+IKF(IKT)+1)
         ENDIF
      ENDDO

      call dfftw_execute(PlanK2ZR)
!
!
      END SUBROUTINE KR_FFTW
!
!
!
! _______________________________________________________________
!
      SUBROUTINE RK_FFTW(UR,VR,WR,UK,FF2,IKN,PlanXR2K,PlanYR2K,PlanZR2K)
!
!      CALLS GRID POINT -> SPECTRAL TRANSFORMS.
!
      USE GLOBAL_PARAMS
      IMPLICIT NONE
!
      INTEGER         :: IKX,IKY,IKZ,IKT,IKN(NKT)
      INTEGER(KIND=8) :: PlanXR2K,PlanYR2K,PlanZR2K
!     
      REAL(rk)     :: UR(NR),VR(NR),WR(NR),INR
      COMPLEX(rk)  :: UK(3*NKT),FF2(NX2*NY*NZ)

      INR = 1._rk/REAL(NR,rk)

!              
      call dfftw_execute(PlanXR2K)
!
!     Unpack and normalise
      DO IKT=1,NKT
         UK(IKT)=FF2(IKN(IKT)+1)*INR
      ENDDO

!
!
      call dfftw_execute(PlanYR2K)
!     Unpack and normalise
!
      DO IKT=1,NKT
         UK(NKT+IKT)=FF2(IKN(IKT)+1)*INR
      ENDDO

!
!
      call dfftw_execute(PlanZR2K)
!     Unpack and normalise
!
      DO IKT=1,NKT
         UK(2*NKT+IKT)=FF2(IKN(IKT)+1)*INR
      ENDDO

!
!
      END SUBROUTINE RK_FFTW
