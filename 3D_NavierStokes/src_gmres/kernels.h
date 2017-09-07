#define nx 128
#define ny 128
#define nz 128
#define nr 2097152
#define nx2 65
#define ktx 42
#define kty 42
#define ktz 42
#define iktx 43
#define ikty 85
#define iktz 85
#define nkt 310675
#define nthreads 512
#define nblocks 160
// Threads should be a multiple of warp size (32) and maximise warps per multiproc to hide register latency (>192)
// Blocks should be a multiple of multiprocs (16) and >100 for scalability

////////////////////////////////////////////////////////////////////////
// Define constant device variables
////////////////////////////////////////////////////////////////////////
__constant__ int d_O2;
__constant__ double d_IN,d_AMPFOR,d_DELT,d_v2,d_KFY;

// Define constant host variables
////////////////////////////////////////////////////////////////////////

const double twopi = 4.0*asin(1.0);

int n2;
double v2,in,tme,tstart,delt,ampfor,kfy;
double *d_kx,*d_ky,*d_kz;

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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void conj(cufftDoubleComplex *ZK,cufftDoubleComplex *ZZ, double *kx){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;

  while(ik < nkt){ 
    if( kx[ik] == 0.0 ){
      int ikt = int(ik/iktx);
      int ikz = int(floor(double(ikt / ikty)));
      int iky = int(ikt - ikz*ikty);

      if( iky == kty ){
	if( ikz == ktz){
	  ZZ[ik].x = 0.0;
	  ZZ[ik].y = 0.0;
	}else if(ikz < ktz){
	int ikzz = iktz-1 - ikz;

	int ikk = (ikzz*ikty + iky)*iktx;
	double ZKX = ZK[ikk].x; 
	double ZKY =-ZK[ikk].y;
	
	ZZ[ik].x =  ZKX;
	ZZ[ik].y =  ZKY;	  
	}else{
	ZZ[ik].x = ZK[ik].x;
	ZZ[ik].y = ZK[ik].y;
	}
      }else if( iky < kty){
	int ikky = ikty-1 -iky;
	
	ikz = iktz-1 - ikz;

	int ikk = (ikz*ikty + ikky)*iktx;
	
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

  while(ik < nkt){
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
      UK[ik].x = (kkz*etY-kky*ztY)*wki;
      UK[ik].y = (kky*ztX-kkz*etX)*wki;
      
      VK[ik].x = (kkx*ztY-kkz*xiY)*wki;
      VK[ik].y = (kkz*xiX-kkx*ztX)*wki;

      WK[ik].x = (kky*xiY-kkx*etY)*wki;
      WK[ik].y = (kkx*etX-kky*xiX)*wki;

      ik += blockDim.x*gridDim.x;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void normFF1(cufftDoubleComplex *F2, cufftDoubleComplex *F1, int *ikN){
  int ik = threadIdx.x + blockIdx.x*blockDim.x;
  
  //normalise D2Z FFT output
  while(ik < nkt){

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

  while(ik < nr){
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
  
  while(ik < nkt){
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

      double NUZN = nuzn[ik];//(1 - wk*d_v2*d_DELT*0.5)/d_DELT;//
      double NU1 = nu1[ik];//d_DELT/(1 + wk*d_v2*d_DELT*0.5);//
      //check if this is the forcing wave number and make appropriate adjustment
      if(LL[ik] == 0){
	// set mask this way to minimise warp divergence
	NU1 = 0.0; 
	wki = 0.0;
      }

      if(kkx==0.0 && kkz==d_KFY && kky==d_KFY) etY += d_AMPFOR;
      if(kkx==0.0 && kkz==d_KFY && kky==d_KFY) ztY -= d_AMPFOR;

      if(kkx==0.0 && kkz==-d_KFY && kky==d_KFY) etY += d_AMPFOR;
      if(kkx==0.0 && kkz==-d_KFY && kky==d_KFY) ztY += d_AMPFOR;
      //      if(kkx==0.0 && kkz==0.0 && kky==d_KFY) ztX += d_AMPFOR;  
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

      // Update spectral velocity coeffs
      UK[ik].x = (kkz*etY-kky*ztY)*wki;
      UK[ik].y = (kky*ztX-kkz*etX)*wki;

      VK[ik].x = (kkx*ztY-kkz*xiY)*wki;
      VK[ik].y = (kkz*xiX-kkx*ztX)*wki;

      WK[ik].x = (kky*xiY-kkx*etY)*wki;
      WK[ik].y = (kkx*etX-kky*xiX)*wki;

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


  while(ik < nkt){
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

      double NU2 =0.5*nu1[ik];// d_DELT/(1 + wk*d_v2*d_DELT*0.5);//

      if(LL[ik] == 0){
	// set mask this way to minimise warp divergence
	NU2 = 0.0;
	wki = 0.0;
      }

      if(kkx==0.0 && kkz==d_KFY && kky==d_KFY) etY += d_AMPFOR;
      if(kkx==0.0 && kkz==d_KFY && kky==d_KFY) ztY -= d_AMPFOR;

      if(kkx==0.0 && kkz==-d_KFY && kky==d_KFY) etY += d_AMPFOR;
      if(kkx==0.0 && kkz==-d_KFY && kky==d_KFY) ztY += d_AMPFOR;

      //      if(kkx==0.0 && kkz==0.0 && kky==d_KFY) ztX += d_AMPFOR;
      //if(kkx==0.0 && kkz==0.0 && kky==4.0) ztX += -2.0;

      // Do corrector step
      xiX = xii[ik].x + NU2*(xiX - RHX[ik].x);
      xiY = xii[ik].y + NU2*(xiY - RHX[ik].y);

      etX = eta[ik].x + NU2*(etX - RHY[ik].x);
      etY = eta[ik].y + NU2*(etY - RHY[ik].y);

      ztX = zet[ik].x + NU2*(ztX - RHZ[ik].x);
      ztY = zet[ik].y + NU2*(ztY - RHZ[ik].y);

      // Update spectral velocity coeffs
      UK[ik].x = (kkz*etY-kky*ztY)*wki;
      UK[ik].y = (kky*ztX-kkz*etX)*wki;

      VK[ik].x = (kkx*ztY-kkz*xiY)*wki;
      VK[ik].y = (kkz*xiX-kkx*ztX)*wki;

      WK[ik].x = (kky*xiY-kkx*etY)*wki;
      WK[ik].y = (kkx*etX-kky*xiX)*wki;

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
