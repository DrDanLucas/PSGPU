      program read
C
C     Program to rewrite mid-planes of 3D binary data
C
      use GLOBAL_PARAMS
      IMPLICIT NONE
      INTEGER :: ikx,iky,ikz,nt,nw,nlx,iopt
C ----------------------------------------------------------------
      REAL(rk) :: ZR(NX,NY,NZ),dy
c
      TWOPI = 4._rk*ASIN(1._rk)
      print*,'Which file?'
      read(5,*) infile
c
      open(12,file=trim(infile),access="STREAM",form='UNFORMATTED')
      open(13,file='vort_xy.dat',form='FORMATTED')
      open(14,file='vort_yz.dat',form='FORMATTED')
      open(15,file='vort_xz.dat',form='FORMATTED')

      read(12) (((ZR(ikx,iky,ikz),ikx=1,NX),iky=1,NY),ikz=1,NZ)

      do iky=1,NY
         write(13,333) (ZR(ikx,iky,NZ/2),ikx=1,NX)
         write(13,*) '           '
         write(14,333) (ZR(NX/2,ikz,iky),ikz=1,NZ)
         write(14,*) '           '
         write(15,333) (ZR(ikx,NY/2,iky),ikx=1,NX)
         write(15,*) '           '
      enddo
      
C
 333  format(1x,E12.5,1x)
      END
C
c
