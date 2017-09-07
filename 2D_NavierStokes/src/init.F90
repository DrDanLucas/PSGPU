!
      SUBROUTINE INIT (Z,TSTART,PERIODin,SHIFTXin,SHIFTYin,NERin)
!
!     INITIALISES STUFF LIKE WAVENUMBERS, INDICES, SPECTRA, PHASES, ETC.
!
      use GLOBAL_PARAMS
      IMPLICIT NONE 
!
      INTEGER         :: IKX,IKY
      INTEGER         :: ntime
      INTEGER         :: jw,kx1,ky1,klag
!
      COMPLEX(rk)     :: Z(IKTX,IKTY)
      COMPLEX(rk)     :: ZN(IKTX,IKTY)
!
      REAL(rk)        :: ranno,TSTART,normRE,normIM
      REAL(rk)        :: KX,KY,WK,EK
      REAL(rk)        :: PHASE,ENSTRO,R1,R2,VK,R3

      REAL(rk)    :: timein,PERIODin,SHIFTXin,SHIFTYin,NERin
      REAL(rk)    :: dummy,NER2
!
      EXTERNAL RANNO
!
      DO IKX=1,IKTX
         KX = REAL(IKX - 1,rk)*alpha
         DO IKY=1,IKTY
            KY = REAL(IKY - KTY - 1,rk)
            L(IKX,IKY)  = 0
            LL(IKX,IKY) = .TRUE.
            WK = KX*KX + KY*KY
            WK = SQRT( WK )
            IF (KX.LT.0)                   LL(IKX,IKY) = .FALSE.
            IF (KX.EQ.0 .AND. KY.LE.0)     LL(IKX,IKY) = .FALSE.
            IF (WK.GT.REAL(KTY,rk)-0.5_rk) LL(IKX,IKY) = .FALSE.
            IF ( LL(IKX,IKY) )             L(IKX,IKY)  = 1
            Z(IKX,IKY)   = CMPLX(0._rk,0._rk,rk)
         ENDDO
      ENDDO
!
      if (irest.ne.0) then
         print*,'          IREST = ',IREST
         print*,'RESTART FROM OUTPUT DATA. '
         open(22,file=trim(infile),form='unformatted',access='stream')
!         read(22) jw,kx1,ky1
         jw =100000
         kx1 = IKTX
         ky1 = IKTY
         print*,'Number of records = ',jw
         print*,'FILE    Iktx,Ikty = ',kx1,ky1
         print*,'FILE      ktx,kty = ',kx1-1,(ky1-1)/2
         print*,'CURRENT Iktx,Ikty = ',IKTX,ikty
         if (kx1.gt.iktx .or. ky1.gt.ikty) then
!            print*,'Do not know how to decrease resolution.'
!            stop
         endif
         if (jw.lt.irest) then
            print*, 'Files not long enough.'
            stop
         endif
         klag = INT(kty) - (ky1-1)/2
!
         Print*,'infile type = ',trim(InfileType)
         do ntime = 1, irest
!
          SELECT CASE (trim(InfileType))
!
            CASE ("A")
               read(22) timein,((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE ("B")  
               IF (ntime .EQ. 1) THEN    ! CASE ("B")
                read(22) timein,((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
               ELSE
                read(22) timein,PERIODin,SHIFTXin,NERin,NER2,                   &
     &                    ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
               ENDIF
!
            CASE ("C")
               read(22) timein,PERIODin,SHIFTXin,NERin,NER2,                    &
     &                   ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE ("D")
               read(22) timein,v2,PERIODin,SHIFTXin,NERin,NER2,                    &
     &                   ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE ("E")
               read(22) timein,dummy,PERIODin,SHIFTXin,NERin,NER2,                    &
     &                   ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE ("G")
               read(22) timein,PERIODin,SHIFTXin,SHIFTYin,NERin,                    &
     &                    ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE ("H")        
               read(22) timein,v2,PERIODin,SHIFTXin,SHIFTYin,NERin,                    & 
     &                    ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE ("I")        
               read(22) timein,dummy,PERIODin,SHIFTXin,SHIFTYin,NERin,                    & 
     &                    ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
            CASE ("J")        
               read(22) timein,dummy,PERIODin,SHIFTXin,SHIFTYin,NERin,dummy,                & 
     &                    ((ZN(ikx,iky+klag),ikx=1,kx1),iky=1,ky1)
!
            CASE DEFAULT
               Print*,'infile type unknown, stopping code'
               STOP
!
          END SELECT
!
          print*,'Just read time= ',timein
         enddo

         DO IKX=1,IKTX
            DO IKY =1,IKTY
               Z(IKX,IKY)=ZN(IKX,IKY)
            ENDDO
         ENDDO

!
         TSTART = TIMEIN
         normIM = SQRT( SUM( AIMAG(Z)**2 , LL ) )
         normRE = SQRT( SUM( REAL(Z)**2 , LL ) )
         IF (irest.NE.0 .AND. WNFLAG.EQ.1) THEN
            print*,'Adding rescale factor ...',scale
         ENDIF
         print*, 'Continuing integration from t = ',timein
         print*, 'normRE = ',normRE,', normIM = ',normIM
         print*, '                 '
         print*, '                 '
         print*, '                 '
      close(22)
      endif
!
!
      ENSTRO = 0.0
!
      DO IKX=1,IKTX
         KX = REAL(IKX-1,rk)*alpha
         DO IKY=1,IKTY
            KY = REAL(IKY-KTY-1,rk)
            WK = SQRT(KX*KX+KY*KY)
!
            IF (L(IKX,IKY).NE.1) THEN
               Z(IKX,IKY)      = CMPLX(0._rk,0._rk,rk)
               NU(IKX,IKY)     = 1000._rk
               CYCLE
            ENDIF
!     
!            Initial Condition is set by setting the desired energy Spectrum
!            ---------------------------------------------------------------- 
!            EK = INITIAL ENERGY    SPECTRUM.
!            VK = INITIAL VORTICITY SPECTRUM.
!            QK =   "      SCALAR      "
!
            IF ( WK.GT.2.5_rk .AND. WK.LE.9.5_rk ) THEN
              EK = 1._rk/WK
            ELSE
              EK = 0.0_rk
            ENDIF
!
            IF (kx.LT.0.1_rk) EK=0._rk
!
!$$$            EK = 1._rk/WK
!
            VK = WK**2._rk * EK
!
            IF (irest.EQ.0) THEN
              PHASE = RANNO(0)
              Z(IKX,IKY) = SQRT(VK/WK) * EXP(ZI*PHASE)
            ENDIF
!
           IF (irest.NE.0 .AND. WNFLAG.EQ.1) THEN
            PHASE = RANNO(0)
            if(KX==1.0 .and. KY==0.0) Z(IKX,IKY)=Z(IKX,IKY)* EXP(ZI*PHASE)
            if(KX==0.0 .and. KY==1.0) Z(IKX,IKY)=Z(IKX,IKY)* EXP(ZI*PHASE)
!            Z(IKX,IKY) = Z(IKX,IKY)*scale
           ENDIF
!
            R1     =  REAL(Z(IKX,IKY),rk)**2._rk
            R2     = AIMAG( Z(IKX,IKY) )**2._rk
            ENSTRO = ENSTRO + R1 + R2
!
            R3 = log10(v2)+2._rk*log10(WK)
            NU(IKX,IKY)     = V1 + 10._rk**R3
         ENDDO
      ENDDO
!
      ENSTRO = ENSTRO*2._rk
!
! -------------------------------------------------------
!     
      IF (irest.EQ.0) THEN
!     NORMALISE AMPLITUDES:
!
         DO IKX=1,IKTX
            KX = REAL(IKX-1,rk)*alpha
            DO IKY=1,IKTY
               KY = REAL(IKY-KTY-1,rk)
               WK = SQRT(KX*KX+KY*KY)
               IF (L(IKX,IKY).EQ.1) THEN
                  Z(IKX,IKY) = Z(IKX,IKY) * SQRT(AMPV/ENSTRO)
               ELSE
                  Z(IKX,IKY) = CMPLX(0._rk,0._rk,rk)
               ENDIF
              IF (lamFLAG.EQ.1 .AND. KX.EQ.0 .AND. KY.EQ.KF) THEN
                 Print*,'adding laminar solution to initial condition'
                 Print*,'IKX = ',IKX,'IKY = ',IKY
!                 IF (IKX.NE.KFX2) Print*,'ERROR: IKX.NE.KFX2' 
!                 IF (IKY.NE.KFY2) Print*,'ERROR: IKY.NE.KFY2'
                 Z(IKX,IKY) = Z(IKX,IKY) - 1._rk/(2._rk*REAL(KF,rk)*v2)
              ENDIF
            ENDDO
         ENDDO
!
      ENDIF
!
      END SUBROUTINE INIT
!
!
!
!
      FUNCTION ranno (i)
!
!     Controls random number generator.
!-----------------------------------------
!   - If argument i.ne.0 it performs
!     initialization with i=seed no.
!   - If argument i.eq.0 it draws a
!     random no.
!-----------------------------------------
      use GLOBAL_PARAMS    
      implicit none
      integer  :: i,junk,ihold,iv(32),iy
      real(rk) :: ranno,ran1
      common /random/ junk,iv,iy
!
      if (i.ne.0) then
        if (i.gt.0) i = - i
        junk  = i
        ranno = (ran1(i)-0.5_rk)*twopi
      else
        junk  = junk - 1
        ihold = junk
        ranno = (ran1(ihold)-0.5_rk)*twopi
      endif
!
      END FUNCTION ranno
!
!
!
      FUNCTION ran1(idum)
      use GLOBAL_PARAMS    
      implicit none
      integer  :: idum,ia,im,iq,ir,ntab,ndiv
      real(rk) :: ran1,am,eps,rnmx
      parameter (ia=16807,im=2147483647,am=1._rk/im,iq=127773,ir=2836,  & 
     &        ntab=32,ndiv=1+(im-1)/ntab,eps=1.2E-7_rk,rnmx=1._rk-eps)
      integer  :: j,k,iv(32),iy,junk
      common /random/ junk,iv,iy
!
      if (idum.le.0.or.iy.eq.0) then
         idum=max(-idum,1)
         do j=ntab+8,1,-1
            k=idum/iq
            idum=ia*(idum-k*iq)-ir*k
            if (idum.lt.0) idum=idum+im
            if (j.le.ntab) iv(j)=idum
         enddo
         iy=iv(1)
      endif
      k=idum/iq
      idum=ia*(idum-k*iq)-ir*k
      if (idum.lt.0) idum=idum+im
      j=1+iy/ndiv
      iy=iv(j)
      iv(j)=idum
      ran1=min(am*iy,rnmx)
!
      END FUNCTION ran1
!
!
      block data
      parameter (ntab=32)
      integer   :: iv(32),iy,junk
      common /random/ junk,iv,iy
      data iv /ntab*0/, iy /0/
      end
