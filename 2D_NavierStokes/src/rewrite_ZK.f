      program read
C
C DIRECT SIMULATION WITH CENTRED DIFFERENCES AND CRANK-NICHOLSON DAMPING.
C
      use GLOBAL_PARAMS
      IMPLICIT NONE
      INTEGER :: ikx,iky,nt,nw,dum,nktx,nkty
C ----------------------------------------------------------------
      REAL(rk) :: bg,time,dummy
      COMPLEX(rk) :: ZO(IKTX,IKTY),ZN(IKTX,IKTY)
c
      print*,'Which output?'
      read(5,*) nw
      print*,'You have selected number ',nw
c
      open(11,file='UPO_Zk.out',access="stream",form='UNFORMATTED')
      open(13,file='Zk_a.dat',form='FORMATTED')
c
c vort.dat is the model output file, vort_a.dat is the rewritten ascii version.
c
c      read(11) dum,nktx,nkty
      do nt=1,10000
         read(11) time,dummy,dummy,dummy,dummy,dummy       
           read(11) ZO
c           read(11) ((ZO(ikx,iky),ikx=1,IKTX),iky=1,IKTY)
          if (nt.eq.nw) then
c           call big(zr,N,N,bg)
             do iky=1,IKTY
c                write(13,333) (ABS(ZO(ikx,iky)),ikx=1,IKTX)
                write(13,333) (ABS(ZO(ikx,iky)),ikx=1,IKTX)
                write(13,*) '           '
           enddo
           print*,'time = ',time
           stop
          endif
c          print*,'nt=',nt
      enddo
C
999   stop
 333  format(1x,E12.5,1x)
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
