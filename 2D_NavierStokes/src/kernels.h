////////////////////////////////////////////////////////////////////////
// Define constant device variables
////////////////////////////////////////////////////////////////////////
__constant__ int d_IKTX, d_IKTY, d_KTY, d_NX, d_NY,d_NX2,d_KFX,d_KFY,d_OR,d_OK,d_O2,d_nsx,d_nsy;
__constant__ double d_IN,d_AMPFOR;

// Define constant host variables
////////////////////////////////////////////////////////////////////////

// Threads should be a multiple of warp size (32) and maximise warps per multiproc to hide register latency (>192)
// Blocks should be a multiple of multiprocs (14) and >100 for scalability
const int nthreads=512;
const int nblocks=140;

int nkt,iktx,ikty,kty,nr,nx,ny,nx2,n2,nOut,nStore,izkout,RecOUTflag,minSTOREouter,iStart,tavgCount;
int np[2];
double v2,in,freq2,tme,tstart,delt,ResidualThreshold,energy,diss,input,ampfor,section,ZPP1,ZPP2;

const int storeSize = 500;
const int normWidth= 42;
const int nWidth2 = normWidth;
const int nNorm=2*normWidth*nWidth2;
const double twopi = 4.0*asin(1.0);
const int NSHIFTX =30;
const int NSHIFTY =4;

// recurrence variables
int iNorm[nNorm];
double shiftX[NSHIFTX],shiftY[NSHIFTY],normZStore[storeSize];
double minRESIDUALouter, minTIMEouter,RecPERIOD,minTIMEinner,normZ;
double minSHIFTXouter,minSHIFTYouter,minSHIFTXinner,minSHIFTYinner;
double *d_kx,*d_ky;
FILE * UPOfile;
FILE * UPOZK;
FILE * stats;
FILE * points;
FILE * avgs;
FILE * vort;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//******************  GPU KERNELS    ***************************
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setFFN(cufftDoubleComplex *V, cufftDoubleComplex *F1, int *ikF){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_O2){
    // Map -Ky values to -Ky+NY and fill trucated wave numbers with zeros
    // Note ikF is set with indexing for this on the fortran side
      int ikk = ikF[ik];
      if(ikk < 0){
	F1[ik].x = 0.0;
	F1[ik].y = 0.0;
      }else{
	F1[ik].x = V[ikk].x;
	F1[ik].y = V[ikk].y;
      }
    
    ik+= blockDim.x*gridDim.x;
  }
}


__global__ void avgVelocity(cufftDoubleReal *UR,cufftDoubleReal *VR,cufftDoubleReal *Urms,cufftDoubleReal *Vrms, int *LL, int tavgCount){
  // update the averages
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OR){

    Urms[ik] = (Urms[ik]*tavgCount + UR[ik]*UR[ik])/(tavgCount+1.0);
    Vrms[ik] = (Vrms[ik]*tavgCount + VR[ik]*VR[ik])/(tavgCount+1.0);

    ik += blockDim.x*gridDim.x;
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
///////////

__global__ void conj(cufftDoubleComplex *ZK, double *kx, double *ky){
  // set the conjugate condition on the axis (might be redundant for certain versions of CUFFT)
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){
    if( kx[ik] == 0.0 && ky[ik] < 0.0){
      int iky = ik/d_IKTX;
      int ikk = (d_IKTY-iky-1)*d_IKTX;

      ZK[ik].x =  ZK[ikk].x;
      ZK[ik].y = -ZK[ikk].y;
    }
    ik += blockDim.x*gridDim.x;
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
///////////

__global__ void subLam(cufftDoubleComplex *ZK,double v2){
  // subtract the laminar profile 
  int ik = (d_KTY+4)*d_IKTX;
  ZK[ik].x += 0.125/v2;

}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
/////////

__global__ void setVelocity(cufftDoubleComplex *ZK,cufftDoubleComplex *UK,cufftDoubleComplex *VK,double *kx,double *ky, int *LL){
  // Kernel to set the spectral velocity coefficients UK and VK for first step
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){

  // Apply mask
    if(LL[ik]==1){
      double kkx = kx[ik];
      double kky = ky[ik];
      
      double wk =kkx*kkx + kky*kky;
      double wki = 1.0/(max(wk,0.001));

      UK[ik].x = -kky*ZK[ik].y*wki;
      UK[ik].y =  kky*ZK[ik].x*wki;
      
      VK[ik].x =  kkx*ZK[ik].y*wki;
      VK[ik].y = -kkx*ZK[ik].x*wki;
      
    }else{
      UK[ik].x = 0.0;
      UK[ik].y = 0.0;
      
      VK[ik].x = 0.0;
      VK[ik].y = 0.0;
    }
    ik += blockDim.x*gridDim.x;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void normFF1(cufftDoubleComplex *F2, cufftDoubleComplex *F1, int *ikN){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;
  
  //normalise D2Z FFT output
  while(ik < 2*d_OK){

    if(ik < d_OK){
      int ikk = ikN[ik];
      F2[ik].x = F1[ikk].x*d_IN;
      F2[ik].y = F1[ikk].y*d_IN;
    }else{
      int ikk = ikN[ik-d_OK]+d_O2;
      F2[ik].x = F1[ikk].x*d_IN;
      F2[ik].y = F1[ikk].y*d_IN;
    }
    ik += blockDim.x*gridDim.x;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void normFFZ(cufftDoubleComplex *F2, cufftDoubleComplex *F1, int *ikN){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  //normalise D2Z FFT output
  while(ik < d_OK){

    int ikk = ikN[ik];
    F2[ik].x = F1[ikk].x*d_IN;
    F2[ik].y = F1[ikk].y*d_IN;
    ik += blockDim.x*gridDim.x;
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void multReal(cufftDoubleReal* Z,cufftDoubleReal* U,cufftDoubleReal* V, cufftDoubleReal* NZ){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;
  // do physical space multiplication (cross product U X Z)

  while(ik < 2*d_OR){
    if(ik < d_OR){
      NZ[ik] = V[ik]*Z[ik];
    }else{
      NZ[ik] = U[ik-d_OR]*Z[ik-d_OR];
    }
    ik += blockDim.x*gridDim.x;
  }

  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void zeroComplex(cufftDoubleComplex *F1){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){
    F1[ik].x = 0.0;
    F1[ik].y = 0.0;

    F1[ik+d_OK].x = 0.0;
    F1[ik+d_OK].y = 0.0;

    ik += blockDim.x*gridDim.x;
  }
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void zeroReal(cufftDoubleReal *F1){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OR){
    F1[ik] = 0.0;

    ik += blockDim.x*gridDim.x;
  }
  
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void subtractReal(cufftDoubleReal *F1,cufftDoubleReal *F2){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OR){
    F1[ik] = F2[ik]-F1[ik]*F1[ik];

    ik += blockDim.x*gridDim.x;
  }
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void preStep(cufftDoubleComplex *Z,cufftDoubleComplex *UK,cufftDoubleComplex *VK,cufftDoubleComplex *NZK,cufftDoubleComplex *RHZ,double *nuzn, double *nu1,double *kx,double *ky,int *LL){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;
  
  while(ik < d_OK){
    if(LL[ik]==1){

      double ZX,ZY;
      double kky = ky[ik];
      double kkx = kx[ik];
      double wk =kkx*kkx + kky*kky;
      double wki = 1.0/(max(wk,0.001));

      // Do final part of 'convol' (CURL)
      double NZKX =  kkx*NZK[ik+d_OK].y + kky*NZK[ik].y;
      double NZKY = -kkx*NZK[ik+d_OK].x - kky*NZK[ik].x;

      // Do predictor step
      RHZ[ik].y = NZKY;
      ZY = nu1[ik]*(nuzn[ik]*Z[ik].y + NZKY);
      
      //this is case A, forcing is (sin4y + A*siny)\hat{x}
      //check if this is a forcing wave number and make appropriate adjustments
      if(kkx == 0.0 && kky == 1.0){
	ZX = nu1[ik]*(nuzn[ik]*Z[ik].x-0.5*d_AMPFOR + NZKX);
	RHZ[ik].x = NZKX-0.5*d_AMPFOR;
      }else if(kkx ==0.0 && kky == 4.0){
	ZX = nu1[ik]*(nuzn[ik]*Z[ik].x -2.0+NZKX);
	RHZ[ik].x = NZKX- 2.0;
      }else{
	ZX = nu1[ik]*(nuzn[ik]*Z[ik].x + NZKX);
	RHZ[ik].x = NZKX;
      }

      // Update spectral velocity coeffs
      UK[ik].x = -kky*ZY*wki;
      UK[ik].y =  kky*ZX*wki;

      VK[ik].x =  kkx*ZY*wki;
      VK[ik].y = -kkx*ZX*wki;

      // Update vorticity array
      Z[ik].x = ZX;
      Z[ik].y = ZY;
    }else{
      UK[ik].x = 0.0;
      UK[ik].y = 0.0;

      VK[ik].x = 0.0;
      VK[ik].y = 0.0;

      Z[ik].x = 0.0;
      Z[ik].y = 0.0;
      
    }

    // reset this array
    NZK[ik].x = 0.0;
    NZK[ik].y = 0.0;

    NZK[ik+d_OK].x = 0.0;
    NZK[ik+d_OK].y = 0.0;

    ik+=blockDim.x*gridDim.x;
    
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void corStep(cufftDoubleComplex *Z,cufftDoubleComplex *UK,cufftDoubleComplex *VK,cufftDoubleComplex *NZK,cufftDoubleComplex *RHZ,double *nu1,double *kx,double *ky, int *LL){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;


  while(ik < d_OK){
  // Apply mask
    if(LL[ik]==1){
      double ZX,ZY;
      double kky = ky[ik];
      double kkx = kx[ik];
      double wk =kkx*kkx + kky*kky;
      double wki = 1.0/(max(wk,0.001));

      // Do final part of 'convol' (CURL)
      double NZKX =  kkx*NZK[ik+d_OK].y + kky*NZK[ik].y;
      double NZKY = -kkx*NZK[ik+d_OK].x - kky*NZK[ik].x;
      
      // Do corrector step
      ZY = Z[ik].y + 0.5*nu1[ik]*(NZKY - RHZ[ik].y);

      //check if this is the forcing wave number and make appropriate adjustments
      if(kkx ==0.0 && kky == 1.0){
      	ZX = Z[ik].x + 0.5*nu1[ik]*(NZKX-0.5*d_AMPFOR  - RHZ[ik].x);
      }else if(kkx == 0.0 && kky ==4.0 ){
      	ZX = Z[ik].x + 0.5*nu1[ik]*(NZKX - 2.0 - RHZ[ik].x);
      }else{
	ZX = Z[ik].x + 0.5*nu1[ik]*(NZKX - RHZ[ik].x);
      }

      // Update spectral velocity coeffs
      UK[ik].x = -kky*ZY*wki;
      UK[ik].y =  kky*ZX*wki;

      VK[ik].x =  kkx*ZY*wki;
      VK[ik].y = -kkx*ZX*wki;

      // Update vorticty array
      Z[ik].x = ZX;
      Z[ik].y = ZY;
 
    }else{
      Z[ik].x = 0.0;
      Z[ik].y = 0.0;

      UK[ik].x = 0.0;
      UK[ik].y = 0.0;

      VK[ik].x = 0.0;
      VK[ik].y = 0.0;
    }

    //reset
    NZK[ik].x = 0.0;
    NZK[ik].y = 0.0;

    NZK[ik+d_OK].x = 0.0;
    NZK[ik+d_OK].y = 0.0;

    ik+=blockDim.x*gridDim.x;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
