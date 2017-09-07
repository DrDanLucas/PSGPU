#define storeSize 500
#define normWidth 8
#define NSHIFTX 30
#define NSHIFTZ 2
#define NSHIFTY 2
////////////////////////////////////////////////////////////////////////
// Define constant device variables
////////////////////////////////////////////////////////////////////////
__constant__ int d_IKTX,d_IKTY,d_IKTZ,d_KTZ,d_NX, d_NY,d_NZ,d_NX2,d_OR,d_OK,d_O2;
__constant__ double d_IN,d_AMPFOR,d_DELT,d_v2,d_KFY;

// Define constant host variables
////////////////////////////////////////////////////////////////////////

// Threads should be a multiple of warp size (32) and maximise warps per multiproc to hide register latency (>192)
// Blocks should be a multiple of multiprocs (14) and >100 for scalability
const int nthreads=512;
const int nblocks=140;
const double twopi = 4.0*asin(1.0);
const int nNorm = 4*normWidth*normWidth*normWidth;

int nkt,iktx,ikty,iktz,nr,nx,ny,nz,nx2,n2,nOut,nStore,izkout,RecOUTflag,minSTOREouter,iStart,tavgCount,NT,rank;
double v2,in,freq2,tme,tstart,delt,ResidualThreshold,energy,diss,input,ampfor,kfy;
double enerAvg, dissAvg, inputAvg,eturbAvg,dturbAvg,enerMin,dissMin,dissMax,enerMax,inputMin,inputMax;

// recurrence variables
int iNorm[nNorm];
double shiftX[NSHIFTX],shiftY[NSHIFTY],shiftZ[NSHIFTZ],normZStore[storeSize];
double minRESIDUALouter, minTIMEouter,RecPERIOD,normZ;
double minSHIFTXouter,minSHIFTYouter,minSHIFTZouter,minSHIFTXinner,minSHIFTYinner,minSHIFTZinner;
double *d_kx,*d_ky,*d_kz,*Energy,*Diss,*Input;
FILE * UPOfile;
FILE * UPOZK;
FILE * sweep;
FILE * ZK;
FILE * stats;
FILE * engy;
FILE * avgs;
FILE * means;
FILE * pdfs;
FILE * xiiStream;
FILE * etaStream;
FILE * zetStream;



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

__global__ void avgVelocity(cufftDoubleReal *UR,cufftDoubleReal *VR,cufftDoubleReal *WR,cufftDoubleReal *Urms,cufftDoubleReal *Vrms,cufftDoubleReal *Wrms, int tavgCount){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OR){

    Urms[ik] = (Urms[ik]*tavgCount + UR[ik]*UR[ik])/(tavgCount+1.0);
    Vrms[ik] = (Vrms[ik]*tavgCount + VR[ik]*VR[ik])/(tavgCount+1.0);
    Wrms[ik] = (Wrms[ik]*tavgCount + WR[ik]*WR[ik])/(tavgCount+1.0);

    ik += blockDim.x*gridDim.x;
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
//////////

__global__ void zeroReal(cufftDoubleReal *F1){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OR){
    F1[ik] = 0.0;

    ik += blockDim.x*gridDim.x;
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
//////////

__global__ void subtractReal(cufftDoubleReal *F1,cufftDoubleReal *F2){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OR){
    F1[ik] = F2[ik]-F1[ik]*F1[ik];

    ik += blockDim.x*gridDim.x;
  }

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void conj(cufftDoubleComplex *ZK,cufftDoubleComplex *ZZ, double *kx){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){ 
    if( kx[ik] == 0.0 ){
      int ikt = int(ik/d_IKTX);
      int ikz = int(floor(double(ikt / d_IKTY)));
      int iky = int(ikt - ikz*d_IKTY);

      if( iky == d_KTZ ){
	if( ikz == d_KTZ){
	  ZZ[ik].x = 0.0;
	  ZZ[ik].y = 0.0;
	}else if(ikz < d_KTZ){
	int ikzz = d_IKTZ-1 - ikz;

	int ikk = (ikzz*d_IKTY + iky)*d_IKTX;
	double ZKX = ZK[ikk].x; 
	double ZKY =-ZK[ikk].y;
	
	ZZ[ik].x =  ZKX;
	ZZ[ik].y =  ZKY;	  
	}else{
	ZZ[ik].x = ZK[ik].x;
	ZZ[ik].y = ZK[ik].y;
	}
      }else if( iky < d_KTZ){
	int ikky = d_IKTY-1 -iky;
	
	ikz = d_IKTZ-1 - ikz;

	int ikk = (ikz*d_IKTY + ikky)*d_IKTX;
	
	double ZKX = ZK[ikk].x; 
	double ZKY =-ZK[ikk].y;
	
	ZZ[ik].x =  ZKX;
	ZZ[ik].y =  ZKY;
	}else{

	ZZ[ik].x = ZK[ik].x;
	ZZ[ik].y = ZK[ik].y;
      }
      }else{
	ZZ[ik].x = ZK[ik].x;
	ZZ[ik].y = ZK[ik].y;
    }
      ik += blockDim.x*gridDim.x;
  }

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setVelocity(cufftDoubleComplex *xii,cufftDoubleComplex *eta,cufftDoubleComplex *zet,cufftDoubleComplex *UK,cufftDoubleComplex *VK,cufftDoubleComplex *WK,double *kx,double *ky,double *kz,int *LL){
  // Kernel to set the spectral velocity coefficients UK and VK for first step
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){
      double kkx = kx[ik];
      double kky = ky[ik];
      double kkz = kz[ik];

      double xiX = xii[ik].x;
      double xiY = xii[ik].y;
      
      double etX = eta[ik].x;
      double etY = eta[ik].y;
      
      double ztX = zet[ik].x;
      double ztY = zet[ik].y;
      

      double wk =kkx*kkx + kky*kky +kkz*kkz;
      double wki = 1.0/(max(wk,0.001));

	// set mask this way to minimise warp divergence
      if(LL[ik] == 0) wki = 0.0;

      // Update spectral velocity coeffs
      UK[ik].x = -(kky*ztY-kkz*etY)*wki;
      UK[ik].y = -(kkz*etX-kky*ztX)*wki;

      VK[ik].x = -(kkz*xiY-kkx*ztY)*wki;
      VK[ik].y = -(kkx*ztX-kkz*xiX)*wki;

      WK[ik].x = -(kkx*etY-kky*xiY)*wki;
      WK[ik].y = -(kky*xiX-kkx*etX)*wki;

      ik += blockDim.x*gridDim.x;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void zeroVorticity(cufftDoubleComplex *xii,cufftDoubleComplex *eta,cufftDoubleComplex *zet){
  // Kernel to set the spectral velocity coefficients UK and VK for first step
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){

      xii[ik].x = 0.0;
      xii[ik].y = 0.0;
      
      eta[ik].x = 0.0;
      eta[ik].y = 0.0;

      zet[ik].x = 0.0;
      zet[ik].y = 0.0;


      ik += blockDim.x*gridDim.x;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setVorticity(cufftDoubleComplex *xii,cufftDoubleComplex *eta,cufftDoubleComplex *zet,double *kx,double *ky,double *kz){
  // Kernel to set the spectral velocity coefficients UK and VK for first step
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){

  if(kx[ik]==0.0 && ky[ik]== 1.0 && kz[ik]== 1.0) xii[ik].x -= 0.5;
  if(kx[ik]==0.0 && ky[ik]== -1.0 && kz[ik]==-1.0) xii[ik].x -= 0.5;
  if(kx[ik]==0.0 && ky[ik]== 1.0 && kz[ik]==-1.0) xii[ik].x +=  0.5;
  if(kx[ik]==0.0 && ky[ik]== -1.0 && kz[ik]==1.0) xii[ik].x +=  0.5;

  ik += blockDim.x*gridDim.x;
  }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void errorCheck(cufftDoubleComplex *xii,cufftDoubleComplex *eta,cufftDoubleComplex *zet,double *kx,double *ky,double *kz){
  // Kernel to set the spectral velocity coefficients UK and VK for first step
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < d_OK){

  if(kx[ik]==0.0 && ky[ik]== 1.0 && kz[ik]== 1.0) xii[ik].x -= 0.5;
  if(kx[ik]==0.0 && ky[ik]== -1.0 && kz[ik]==-1.0) xii[ik].x -= 0.5;
  if(kx[ik]==0.0 && ky[ik]== 1.0 && kz[ik]==-1.0) xii[ik].x +=  0.5;
  if(kx[ik]==0.0 && ky[ik]== -1.0 && kz[ik]==1.0) xii[ik].x +=  0.5;

  ik += blockDim.x*gridDim.x;
  }

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void normFF1(cufftDoubleComplex *F2, cufftDoubleComplex *F1, int *ikN){
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

__global__ void multReal(cufftDoubleReal* xii,cufftDoubleReal* eta,cufftDoubleReal* zet,cufftDoubleReal* U,cufftDoubleReal* V,cufftDoubleReal* W){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;
  // do physical space multiplication (cross product U X Z)

  while(ik < d_OR){
    double TU,TV,TW,TE,TZ,TX,NX,NY;
    TV = V[ik];
    TU = U[ik];
    TW = W[ik];

    TX = xii[ik];
    TE = eta[ik];
    TZ = zet[ik];

    NX = TV*TZ-TW*TE;
    NY = TW*TX-TU*TZ;
    TZ = TU*TE-TV*TX;
    
    U[ik]=-NX;
    V[ik]=-NY;
    W[ik]=-TZ;
    ik += blockDim.x*gridDim.x;
  }

  
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void preStep(cufftDoubleComplex *xii,cufftDoubleComplex *eta,cufftDoubleComplex *zet,cufftDoubleComplex *UK,cufftDoubleComplex *VK,cufftDoubleComplex *WK,cufftDoubleComplex *RHX,cufftDoubleComplex *RHY,cufftDoubleComplex *RHZ,double *nuzn, double *nu1,double *kx,double *ky,double *kz, int *LL){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;
  
  while(ik < d_OK){
      double kky = ky[ik];
      double kkx = kx[ik];
      double kkz = kz[ik];
      double UKX = UK[ik].x;
      double UKY = UK[ik].y;
      double VKX = VK[ik].x;
      double VKY = VK[ik].y;
      double WKX = WK[ik].x;
      double WKY = WK[ik].y;

      double wk =kkx*kkx + kky*kky +kkz*kkz;
      double wki = 1.0/(max(wk,0.001));

      // Do final part of 'convol' (CURL)
      double xiX =  kky*WKY - kkz*VKY;
      double xiY =  kkz*VKX - kky*WKX;

      double etX =  kkz*UKY - kkx*WKY;
      double etY =  kkx*WKX - kkz*UKX;

      double ztX =  kkx*VKY - kky*UKY;
      double ztY =  kky*UKX - kkx*VKX;

      double NUZN = nuzn[ik];//(1 - wk*d_v2*d_DELT*0.5)/d_DELT;
      double NU1 = LL[ik]*nu1[ik];//d_DELT/(1 + wk*d_v2*d_DELT*0.5);
      // set mask this way to minimise warp divergence

      //check if this is the forcing wave number and make appropriate adjustment

      if(kkx==0.0 && kkz==d_KFY && kky==d_KFY) xiX += d_AMPFOR;
      if(kkx==0.0 && kkz==-d_KFY && kky==d_KFY) xiX -= d_AMPFOR;

      // Do predictor step

      RHX[ik].x = xiX;
      RHX[ik].y = xiY;
      xiX = NU1*(NUZN*xii[ik].x + xiX);
      xiY = NU1*(NUZN*xii[ik].y + xiY);

      RHY[ik].x = etX;
      RHY[ik].y = etY;
      etX = NU1*(NUZN*eta[ik].x + etX);
      etY = NU1*(NUZN*eta[ik].y + etY);
      
      RHZ[ik].x = ztX;
      RHZ[ik].y = ztY;
      ztX = NU1*(NUZN*zet[ik].x + ztX);
      ztY = NU1*(NUZN*zet[ik].y + ztY);

      // Update spectral velocity coeff
      UK[ik].x = -(kky*ztY-kkz*etY)*wki;
      UK[ik].y = -(kkz*etX-kky*ztX)*wki;

      VK[ik].x = -(kkz*xiY-kkx*ztY)*wki;
      VK[ik].y = -(kkx*ztX-kkz*xiX)*wki;

      WK[ik].x = -(kkx*etY-kky*xiY)*wki;
      WK[ik].y = -(kky*xiX-kkx*etX)*wki;

      // Update vorticity arrays
      xii[ik].x = xiX;
      xii[ik].y = xiY;

      eta[ik].x = etX;
      eta[ik].y = etY;

      zet[ik].x = ztX;
      zet[ik].y = ztY;


    ik+=blockDim.x*gridDim.x;
    
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void corStep(cufftDoubleComplex *xii,cufftDoubleComplex *eta,cufftDoubleComplex *zet,cufftDoubleComplex *UK,cufftDoubleComplex *VK,cufftDoubleComplex *WK,cufftDoubleComplex *RHX,cufftDoubleComplex *RHY,cufftDoubleComplex *RHZ,double *nuzn, double *nu1,double *kx,double *ky,double *kz, int *LL){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;


  while(ik < d_OK){
      double kky = ky[ik];
      double kkx = kx[ik];
      double kkz = kz[ik];
      double UKX = UK[ik].x;
      double UKY = UK[ik].y;
      double VKX = VK[ik].x;
      double VKY = VK[ik].y;
      double WKX = WK[ik].x;
      double WKY = WK[ik].y;

      double wk =kkx*kkx + kky*kky +kkz*kkz;
      double wki = 1.0/(max(wk,0.001));

      // Do final part of 'convol' (CURL)
      double xiX =  kky*WKY - kkz*VKY;
      double xiY =  kkz*VKX - kky*WKX;

      double etX =  kkz*UKY - kkx*WKY;
      double etY =  kkx*WKX - kkz*UKX;

      double ztX =  kkx*VKY - kky*UKY;
      double ztY =  kky*UKX - kkx*VKX;

      double NU2 = 0.5*nu1[ik];//d_DELT/(1 + wk*d_v2*d_DELT*0.5);

      if(kkx==0.0 && kkz==d_KFY && kky==d_KFY) xiX += d_AMPFOR;
      if(kkx==0.0 && kkz==-d_KFY && kky==d_KFY) xiX -= d_AMPFOR;

      // Do corrector step
      // set mask this way to minimise warp divergence
      xiX = LL[ik]*(xii[ik].x + NU2*(xiX - RHX[ik].x));
      xiY = LL[ik]*(xii[ik].y + NU2*(xiY - RHX[ik].y));

      etX = LL[ik]*(eta[ik].x + NU2*(etX - RHY[ik].x));
      etY = LL[ik]*(eta[ik].y + NU2*(etY - RHY[ik].y));
      
      ztX = LL[ik]*(zet[ik].x + NU2*(ztX - RHZ[ik].x));
      ztY = LL[ik]*(zet[ik].y + NU2*(ztY - RHZ[ik].y));
 
      // Update spectral velocity coeffs
      UK[ik].x = -(kky*ztY-kkz*etY)*wki;
      UK[ik].y = -(kkz*etX-kky*ztX)*wki;

      VK[ik].x = -(kkz*xiY-kkx*ztY)*wki;
      VK[ik].y = -(kkx*ztX-kkz*xiX)*wki;

      WK[ik].x = -(kkx*etY-kky*xiY)*wki;
      WK[ik].y = -(kky*xiX-kkx*etX)*wki;

      // Update vorticity arrays
      xii[ik].x = xiX;
      xii[ik].y = xiY;

      eta[ik].x = etX;
      eta[ik].y = etY;

      zet[ik].x = ztX;
      zet[ik].y = ztY;

      RHX[ik].x = 0.0;
      RHX[ik].y = 0.0;

      RHY[ik].x = 0.0;
      RHY[ik].y = 0.0;

      RHZ[ik].x = 0.0;
      RHZ[ik].y = 0.0;

    ik+=blockDim.x*gridDim.x;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
