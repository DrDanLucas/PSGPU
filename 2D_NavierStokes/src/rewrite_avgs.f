      program read
C
C DIRECT SIMULATION WITH CENTRED DIFFERENCES AND CRANK-NICHOLSON DAMPING.
C
      use GLOBAL_PARAMS
      IMPLICIT NONE
      INTEGER :: ikx,iky,nt,nw,iopt
C ----------------------------------------------------------------
      REAL(rk) :: ZR(NX,NY),bg,time,y
c
      TWOPI = 4._rk*ASIN(1._rk)
      print*,'Which output ubar (1),vbar (2),Urms (3),Vrms (4)?'
      read(5,*) nw
      print*,'You have selected number ',nw

      print*,'x average? yes: (1), no: (2)?'
      read(5,*) iopt
c
      open(12,file='avgs.dat',access="stream",form='UNFORMATTED')
      open(13,file='avg.dat',form='FORMATTED')
c
c vort.dat is the model output file, vort_a.dat is the rewritten ascii version.
c
      do nt=1,10000
          read(12) time,((ZR(ikx,iky),ikx=1,NX),iky=1,NY)
c          read(12,*) ((ZR(ikx,iky),ikx=1,N),iky=1,N)
          if (nt.eq.nw) then
c           call big(zr,N,N,bg)
           do iky=1,NY
              y = iky*twopi/NY
              if(iopt .eq. 1) then
                 write(13,333) y,SUM(ZR(1:NX,iky))
              else
                 write(13,'(1x,E12.5,1x)') (ZR(ikx,iky),ikx=1,NX)
                 write(13,*) '           '
              endif
           enddo
           print*,'time = ',time
           stop
          endif
c          print*,'nt=',nt
      enddo
C
999   stop
 333  format(2(1x,E12.5,1x))
      END
C
c
c
      subroutine big(zr,N,M,bg)
c
c  Just for scaling the data to make graphs prettier...
c
      implicit none
      integer i,j,N,i2,M
      real    zr(N,M),zmax,zmin,bg
c
        bg=4.
c
        zmax=+1.01*sqrt(bg)
        zmin=-zmax
c
      do j=1,M
      do i=1,N
        if (zr(i,j).ge.0.) zr(i,j)=sqrt(zr(i,j))
        if (zr(i,j).lt.0.) zr(i,j)=-sqrt(-zr(i,j))
        if (zr(i,j).lt.zmin) zr(i,j)=zmin
        if (zr(i,j).gt.zmax) zr(i,j)=zmax
      enddo
      enddo
c
      zr(2,2)=zmax
      zr(1,1)=zmin
c
      return
      end
c
