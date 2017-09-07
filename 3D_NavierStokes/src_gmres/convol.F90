!
      SUBROUTINE CONVOL(ZK,FF2,UK,UR,VR,WR,XIR,ETR,ZTR,KX,KY,KZ,IKF,IKN)
!
!     CALCULATES CONVOLUTION SUMS, CALLS FFT'S, ETC.
!
      USE GLOBAL_PARAMS
      IMPLICIT NONE 
!
      INTEGER                        :: IKX,IKY,IKT,I,J
      INTEGER                        :: IKF(NX2*NY*NZ),IKN(NKT)
!
      REAL(rk)                       :: WK,WEIGHT,WKI,TX,TY,TZ
      REAL(rk), DIMENSION(NR)        :: UR,VR,WR,XIR,ETR,ZTR
      REAL(rk), DIMENSION(NKT)       :: KX,KY,KZ
!
      COMPLEX(rk), DIMENSION(3*NKT) :: ZK,UK
      COMPLEX(rk)                   :: TA,TB,TC,FF2(NX2*NY*NZ)
!
      DO IKT = 1, NKT
            WKI=1._rk/MAX( KX(IKT)*KX(IKT)+KY(IKT)*KY(IKT)+KZ(IKT)*KZ(IKT),0.001_rk) 

            UK(IKT)      = -L(IKT)*ZI*WKI*(KZ(IKT)*ZK(NKT+IKT)-KY(IKT)*ZK(2*NKT+IKT))
            UK(NKT+IKT)  = -L(IKT)*ZI*WKI*(KX(IKT)*ZK(2*NKT+IKT)-KZ(IKT)*ZK(IKT))
            UK(2*NKT+IKT)= -L(IKT)*ZI*WKI*(KY(IKT)*ZK(IKT)-KX(IKT)*ZK(NKT+IKT))
      ENDDO
!
      CALL KR_FFTW(ZK,XIR,ETR,ZTR,FF2,IKF,PlanK2XIR,PlanK2ETR,PlanK2ZTR)
      CALL KR_FFTW(UK,UR,VR,WR,FF2,IKF,PlanK2UR,PlanK2VR,PlanK2WR)

!
      DO I=1,NR
         TX = VR(I)*ZTR(I)-WR(I)*ETR(I)
         TY = WR(I)*XIR(I)-UR(I)*ZTR(I)
         TZ = UR(I)*ETR(I)-VR(I)*XIR(I)

         UR(I)=TX
         VR(I)=TY
         WR(I)=TZ
      ENDDO
!
      CALL RK_FFTW(UR,VR,WR,UK,FF2,IKN,PlanUR2K,PlanVR2K,PlanWR2K)
!
!
!
      DO IKT=1,NKT
         TA = -ZI*(KZ(IKT)*UK(NKT+IKT  )-KY(IKT)*UK(2*NKT+IKT))
         TB = -ZI*(KX(IKT)*UK(2*NKT+IKT)-KZ(IKT)*UK(IKT      ))
         TC = -ZI*(KY(IKT)*UK(IKT      )-KX(IKT)*UK(NKT+IKT  ))

         UK(IKT      ) = TA*L(IKT)
         UK(NKT+IKT  ) = TB*L(IKT)
         UK(2*NKT+IKT) = TC*L(IKT)
      ENDDO
!
      END SUBROUTINE CONVOL
