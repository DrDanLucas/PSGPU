!
      SUBROUTINE TIMEDERIV(ZO,RHZ,KFT1,KFT2,FF2,UK,UR,VR,WR,XIR,ETR,ZTR,  & 
     &                     KX,KY,KZ,NU,IKF,IKN)
!
!     CALCULATED RHS OF THE N-S EQN, I.E. DU/DT
!
      USE GLOBAL_PARAMS
      IMPLICIT NONE 
!
      INTEGER  :: KFT1,KFT2,IKT
      INTEGER  :: IKF(NX2*NY*NZ),IKN(NKT)
!
      REAL(rk), DIMENSION(NKT) :: KX,KY,KZ,NU
      REAL(rk), DIMENSION(NR) :: UR,VR,WR,XIR,ETR,ZTR
!
      COMPLEX(rk), DIMENSION(3*NKT) :: ZO,RHZ,UK
      COMPLEX(rk), DIMENSION(NX2*NY*NZ) :: FF2
!
!
!---------------------------------------------------------
!     GET NONLINEAR TERM 
      CALL CONVOL(ZO,FF2,UK,UR,VR,WR,XIR,ETR,ZTR,KX,KY,KZ,IKF,IKN)
!---------------------------------------------------------
!
!     ADD FORCING TERM
        UK(NKT+KFT1) = UK(NKT+KFT1) + AMPFOR*ZI
        UK(2*NKT+KFT1) = UK(2*NKT+KFT1) - AMPFOR*ZI
        UK(NKT+KFT2) = UK(NKT+KFT2) + AMPFOR*ZI
        UK(2*NKT+KFT2) = UK(2*NKT+KFT2) + AMPFOR*ZI

!     ADD THE DISSIPATION TO THE NON-TRUNCATED MODES
        DO IKT = 1,NKT
           IF (LL(IKT))THEN 
              RHZ(IKT) =  - NU(IKT)*ZO(IKT)  + UK(IKT) 
              RHZ(NKT+IKT) =  - NU(IKT)*ZO(NKT+IKT)  + UK(NKT+IKT) 
              RHZ(2*NKT+IKT) =  - NU(IKT)*ZO(2*NKT+IKT)  + UK(2*NKT+IKT) 
           ELSE
              RHZ(IKT) = CMPLX(0._rk,0._rk,rk)
              RHZ(NKT+IKT) = CMPLX(0._rk,0._rk,rk)
              RHZ(2*NKT+IKT) = CMPLX(0._rk,0._rk,rk)
           ENDIF
        ENDDO
!
      END SUBROUTINE TIMEDERIV
