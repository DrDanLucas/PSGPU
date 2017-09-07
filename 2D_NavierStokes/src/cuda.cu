#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"
///////////////////////////////////////////////////////////////////////////////

extern "C" void set_gpu_(int *go)
{
  int            dev, deviceCount;
  cudaDeviceProp devProp;

  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("cutil error: no devices supporting CUDA\n");
    exit(-1);
  }

  checkCudaErrors(cudaGetDevice(&dev));
  //  checkCudaErrors(cudaSetDevice());
  checkCudaErrors(cudaGetDeviceProperties(&devProp,dev));
  printf("\n Using CUDA device %d: %s\n\n", dev,devProp.name);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////// \
////////////////////////

extern "C" void KR_FFT_ALL(cufftDoubleComplex *d_ZK, cufftDoubleComplex *d_UK,cufftDoubleComplex *d_VK, cufftDoubleReal*d_ZR, cufftDoubleReal*d_UR, cufftDoubleReal*d_VR,cufftHandle PlanZ2D,int *d_ikF){
  cufftDoubleComplex *d_FF;

  //This version does each variable separately (FFT for each Z, U and V rather than one batched FFT)

  checkCudaErrors(cudaMalloc((void**)&d_FF,sizeof(cufftDoubleComplex)*(ny)*(nx2)));

  conj<<<nblocks,nthreads>>>(d_ZK,d_kx,d_ky);
  setFFN<<<nblocks,nthreads>>>(d_ZK,d_FF,d_ikF);  // Pad truncated wave numbers

  (cufftExecZ2D(PlanZ2D,d_FF,d_ZR));  // Do Z FFT

  conj<<<nblocks,nthreads>>>(d_UK,d_kx,d_ky);
  setFFN<<<nblocks,nthreads>>>(d_UK,d_FF,d_ikF);  // Pad truncated wave numbers

  (cufftExecZ2D(PlanZ2D,d_FF,d_UR));  // Do U FFT

  conj<<<nblocks,nthreads>>>(d_VK,d_kx,d_ky);
  setFFN<<<nblocks,nthreads>>>(d_VK,d_FF,d_ikF);  // Pad truncated wave numbers

  (cufftExecZ2D(PlanZ2D,d_FF,d_VR));  // Do V FFT

  checkCudaErrors(cudaFree(d_FF));

 }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" double computeNorm(double2 *Zstore,int iStore){
  
  double normZ=0.0;
  for(int i=0; i<nNorm; i++){
    int ik = iNorm[i]+iStore*nkt;
    normZ += Zstore[ik].x*Zstore[ik].x + Zstore[ik].y*Zstore[ik].y;
  }
  return normZ;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void computeStats(double2 *Zstore,double2 *ZKtavg,double *kx, double *ky,int *LL,int iStore){
  
  double eturb = 0.0;
  double dturb = 0.0;

  energy = 0.0;
  diss = 0.0;
  input = 0.0;

  for(int i=0; i<nkt; i++){
    if(LL[i] ==1 ){
      int ik = i + iStore*nkt;
      double wk = max(kx[i]*kx[i] + ky[i]*ky[i],0.001);
      double VZ = Zstore[ik].x*Zstore[ik].x + Zstore[ik].y*Zstore[ik].y;
      diss += VZ;
      energy += VZ/wk;
      ZKtavg[i].x = (ZKtavg[i].x*tavgCount + Zstore[ik].x)/(tavgCount+1.0);
      ZKtavg[i].y = (ZKtavg[i].y*tavgCount + Zstore[ik].y)/(tavgCount+1.0);
      VZ = ZKtavg[i].x*ZKtavg[i].x + ZKtavg[i].y*ZKtavg[i].y;
      eturb += VZ/wk;
      dturb += VZ;
      if(kx[i] == 0.0 && ky[i] ==4.0 ){
        input += - Zstore[ik].x/4.0;
	//	fprintf(points,"%.15e %.15e \n",Zstore[ik].x,Zstore[ik].y);
      }
    }
  }
  tavgCount++;
  diss *= 2.0*v2;
  dturb *= 2.0*v2;
  fprintf(stats,"%.15e %.15e %.15e %.15e %.15e %.15e \n",tme,energy,eturb,diss,dturb,input);
  
  double s2 = diss - input;

  double PP1 = Zstore[(kty+1)*iktx+iStore*nkt].x;
  double PP2 = Zstore[(kty+2)*iktx+iStore*nkt].x;

  if(section < 0.0 && s2 >0.0){
    double w1 = s2/(s2-section);
    double w2 = -section/(s2-section);

    fprintf(points,"%.15e %.15e \n",w2*PP1+w1*ZPP1,w2*PP2+w1*ZPP2);
  }
  fflush(points);
  fflush(stats);
  section = s2;

  ZPP1 = PP1;
  ZPP2 = PP2;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void computeZdash(double2 *Zstore,
                             double2 *Zdash,
                             double *kx,
                             double *ky,
			     int iStore
                             ){

  for(int iShiftY=0; iShiftY < NSHIFTY; iShiftY++){
    for(int iShiftX=0; iShiftX < NSHIFTX; iShiftX++){

      for(int i=0; i<nNorm; i++){
        int ik =iNorm[i];
        int ikk =ik+iStore*nkt;
        int iks = i + nNorm*(iShiftX+NSHIFTX*iShiftY);

        double kkx = kx[ik];
        double kky = ky[ik];
        double shift = kkx*shiftX[iShiftX]+kky*shiftY[iShiftY];

        Zdash[iks].x = cos(shift)*Zstore[ikk].x+sin(shift)*Zstore[ikk].y;
        Zdash[iks].y = cos(shift)*Zstore[ikk].y-sin(shift)*Zstore[ikk].x;
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////

extern "C" void RK_FFT(cufftDoubleComplex *d_ZK, cufftDoubleReal*d_ZR,cufftHandle PlanD2Z,int *d_ikN){
  cufftDoubleComplex *d_FF;

  //Doing a batch of 2 2D FFTs so that now NZK stores the spectral UK and NZK from the old convol in that order.

  checkCudaErrors(cudaMalloc((void**)&d_FF,sizeof(cufftDoubleComplex)*2*(ny)*(nx2)));

  (cufftExecD2Z(PlanD2Z,d_ZR,d_FF));

  normFF1<<<nblocks,nthreads>>>(d_ZK,d_FF,d_ikN);  // normalise output

  checkCudaErrors(cudaFree(d_FF));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////

extern "C" void RK_FFT_Z(cufftDoubleComplex *d_ZK, cufftDoubleReal*d_ZR,cufftHandle PlanD2Z,int *d_ikN){
  cufftDoubleComplex *d_FF;

  //Doing a batch of 2 2D FFTs so that now NZK stores the spectral UK and NZK from the old convol in that order.

  checkCudaErrors(cudaMalloc((void**)&d_FF,sizeof(cufftDoubleComplex)*(ny)*(nx2)));

  (cufftExecD2Z(PlanD2Z,d_ZR,d_FF));

  normFFZ<<<nblocks,nthreads>>>(d_ZK,d_FF,d_ikN);  // normalise output

  checkCudaErrors(cudaFree(d_FF));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void recurrence_check(double2 *Zstore,
				 double *kx,
				 double *ky,
				 int NT,
				 int nCopy
				 ){

  double minSx,minSy,resPrev,timeUNDER;
  double minRESIDUALinner = 10000.0;
  double minPERIOD = 10000.0;
  double res = 0.0;
  double normFN = 0.0;
  double normFNN = 0.0;
  int minPflag =1;
  int minSTOREinner =0;
  int minCOUNTinner =0;
  double2 Zdash[nNorm*NSHIFTX*NSHIFTY];
  
  computeZdash(Zstore,Zdash,kx,ky,nCopy);
  tme=tstart+(NT-nOut)*delt;
  for(int iCount=(nStore-1); iCount>=1; iCount--){ // note this loop is nStore-1 long
    int iStore;
    double PERIODinner = ((nStore-iCount)*nOut)*(delt);

    // increment over the previous states
    // if nStore = storeSize then we cycle the starting state, iStart
    // this construction below permutes iStore over the storeSize
    iStore = iCount +iStart;
    iStore -= floor(iStore/storeSize)*storeSize;

    //        To save time: when period .gt. 20
    //       only compare evey other previous state.
    if( PERIODinner>20.0 && abs(PERIODinner-freq2*floor(PERIODinner/freq2+0.5))>0.0001)continue;
	  
    resPrev = res;
    res = 10000.0;
    
    for(int iShiftY=0; iShiftY < NSHIFTY; iShiftY++){
      for(int iShiftX=0; iShiftX < NSHIFTX; iShiftX++){
	int is = nNorm*(iShiftX+NSHIFTX*iShiftY);
	double nZdash = 0.0;
	normFNN = 0.0;
	for(int i=0; i<nNorm;i++){
	  int ik = iNorm[i]+iStore*nkt;
	  normFNN -= 2.0*(Zdash[is+i].x*Zstore[ik].x + Zdash[is+i].y*Zstore[ik].y);
	  nZdash += Zdash[is+i].x*Zdash[is+i].x + Zdash[is+i].y*Zdash[is+i].y;
	}
	normFN = normFNN + nZdash + normZStore[iStore];	    
	
	if(sqrt(normFN/normZStore[nCopy]) < res){
	  res = sqrt(normFN/normZStore[nCopy]);
	  minSx = shiftX[iShiftX];
	  minSy = shiftY[iShiftY];
	}
      }
    }
    // Find first turning point as T increases                                                        
    if(res > resPrev && minPflag==1){
      minPERIOD = PERIODinner;
    }else{
      minPflag = 0;
    }
    // If new lower residual is found then store parameters                                           
    if(res < minRESIDUALinner && PERIODinner > minPERIOD){
      minRESIDUALinner = res;
      minSTOREinner =  iStore;
      minCOUNTinner =  nStore - iCount;
      minSHIFTXinner = minSx;
      minSHIFTYinner = minSy;
    }
    
  }
  
  // When min Residual first drops below threshold start new output sequence
  if( minRESIDUALinner < ResidualThreshold && RecOUTflag == 0 ){
    printf("------------------------------------------------\n");
    printf("Time = %e UPO Guess found \n",tme);
    minRESIDUALouter = ResidualThreshold;
    minTIMEinner = tme + minCOUNTinner*nOut*delt;
    RecOUTflag = 1;
  }
  
  // Locate minimum residual for current output sequence and store corresponding data
  if( minRESIDUALinner < minRESIDUALouter && RecOUTflag==1 ){
    minRESIDUALouter = minRESIDUALinner;
    minSTOREouter =  minSTOREinner;
    minTIMEouter = tme + minCOUNTinner*nOut*delt;
    minSHIFTXouter = minSHIFTXinner;
    minSHIFTYouter = minSHIFTYinner;
    RecPERIOD = ((minCOUNTinner)*nOut)*(delt);
  }
  
  // Output best UPO guess at end of current output sequence                                           
  if( minRESIDUALinner > ResidualThreshold && RecOUTflag==1 ){
    RecOUTflag = 0;
    
    // Also compute time spent under residual threshold, normalised by period
    minTIMEinner = tme + minCOUNTinner*nOut*delt- minTIMEinner; 
    timeUNDER = sqrt(minTIMEinner*minTIMEinner/(RecPERIOD*RecPERIOD));

    fwrite(&minTIMEouter,sizeof(double),1,UPOZK);
    fwrite(&v2,sizeof(double),1,UPOZK);
    fwrite(&RecPERIOD,sizeof(double),1,UPOZK);
    fwrite(&minSHIFTXouter,sizeof(double),1,UPOZK);
    fwrite(&minSHIFTYouter,sizeof(double),1,UPOZK);
    fwrite(&minRESIDUALouter,sizeof(double),1,UPOZK);
    fwrite(&timeUNDER,sizeof(double),1,UPOZK);
    fwrite(Zstore+minSTOREouter*nkt,sizeof(double2),nkt,UPOZK);
    izkout = izkout +1;
    printf("Wrote to UPO_Zk.out number %d at time %e \n",izkout,tme);
    
    fprintf(UPOfile,"%d %e %e %e %e %e %e\n",izkout, minTIMEouter,RecPERIOD,minSHIFTXouter,minSHIFTYouter,minRESIDUALouter,timeUNDER);
  }
  fflush(UPOfile);
  fflush(UPOZK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
////////////////////////

extern "C" void timestep_cuda_(double2 *Z,
			       double *ZR,
			       double *nuzn,
			       double *nu1,
			       double *kx,
			       double *ky,
			       double *Time,
			       double *TSTART,
			       double *AMPFOR,
			       double *DELT,
			       double *alpha,
			       double *V2,
			       double *resThresh,
			       int *ikF,
			       int *ikN,
			       int * IKTX,
			       int * IKTY,
			       int * KTY,
			       int * NX,
			       int * NY,
			       int *NSTOP,
			       int *NOUT,
			       int *NOUT2,
			       int *LC,
			       int *RecFLAG,
			       int *statsFLAG)
{
  
  // Define CPU variables
  printf("\n In timestep_cuda \n");
  //Place parameters into global holders
  iktx = *IKTX;
  ikty = *IKTY;
  kty = *KTY;
  ny = *NY;
  nx = *NX;
  nkt = (*IKTX)*(*IKTY);
  nr  = (*NY)*(*NX);
  nx2 = (*NX)/2 +1;
  n2 = nx2*(*NY);
  nOut = *NOUT;
  v2 = *V2;
  in = 1.0/nr;
  tme = *Time;
  tstart = *TSTART;
  delt = *DELT;
  ResidualThreshold = *resThresh;
  np[0] = *NY;
  np[1] = *NX; 
  ampfor = *AMPFOR;
  int nVort = *NOUT2;
  //  if(*NSTOP > 399) nVort = *NSTOP/200;
  section = 1.0;
  ZPP1 = 0.0;
  ZPP2 = 0.0;

  // recurrence variables
  nStore = 0;
  izkout=0;
  RecOUTflag =0;
  minSTOREouter =0;
  freq2 = 2*nOut**DELT;
  tavgCount =0;
  int nCopy;
  double2 *Zstore, *ZKtavg;
  size_t avail,total;
 
  // Define global device variables
  // For LL couldn't read in fortran logical to C bool
  int *d_LL;
  int *d_ikF, *d_ikN;
  cufftDoubleComplex *d_Z, *d_UK, *d_VK, *d_NZK, *d_RHZ;
  cufftDoubleReal*d_ZR,*d_NZR, *d_UR,*d_VR;
  double *d_nuzn,*d_nu1;//, *d_kx,*d_ky;
  double *d_Urms,*d_Vrms;
  cufftHandle PlanZ2D,PlanBatchD2Z,PlanD2Z;

  // Allocate global memory on GPU. (Constant memory does not need allocating) 	
  checkCudaErrors(cudaMalloc((void**)&d_Z,sizeof(cufftDoubleComplex)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_UK,sizeof(cufftDoubleComplex)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_VK,sizeof(cufftDoubleComplex)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_NZK,sizeof(cufftDoubleComplex)*2*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_RHZ,sizeof(cufftDoubleComplex)*nkt));
  
  checkCudaErrors(cudaMalloc((void**)&d_ZR,sizeof(cufftDoubleReal)*nr));
  checkCudaErrors(cudaMalloc((void**)&d_UR,sizeof(cufftDoubleReal)*nr));
  checkCudaErrors(cudaMalloc((void**)&d_VR,sizeof(cufftDoubleReal)*nr));
  checkCudaErrors(cudaMalloc((void**)&d_NZR,sizeof(cufftDoubleReal)*2*nr));
  
  checkCudaErrors(cudaMalloc((void**)&d_nuzn,sizeof(double)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_nu1,sizeof(double)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_kx,sizeof(double)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_ky,sizeof(double)*nkt));
  checkCudaErrors(cudaMalloc((void**)&d_ikF,sizeof(int)*n2));
  checkCudaErrors(cudaMalloc((void**)&d_ikN,sizeof(int)*n2));
  checkCudaErrors(cudaMalloc((void**)&d_LL,sizeof(int)*nkt));
  
  // Copy state data to GPU global memory 
 checkCudaErrors(cudaMemcpy(d_Z,Z,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyHostToDevice));
 // checkCudaErrors(cudaMemcpy(d_ZR,ZR,sizeof(cufftDoubleReal)*nr,cudaMemcpyHostToDevice));
  
  // Copy constant parameters to GPU constant memory
  checkCudaErrors(cudaMemcpyToSymbol(d_IN,&in,sizeof(double)));
  checkCudaErrors(cudaMemcpyToSymbol(d_AMPFOR,AMPFOR,sizeof(double)));
  checkCudaErrors(cudaMemcpyToSymbol(d_IKTX,IKTX,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_IKTY,IKTY,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_KTY,KTY,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_NX,NX,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_NX2,&nx2,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_NY,NY,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_OR,&nr,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_OK,&nkt,sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(d_O2,&n2,sizeof(int)));

  //Set up various arrays to enable generic kernel calls
  //i.e. calculate indexing for padding either side of FFTs, wavenumber arrays, mask, and timestep arrays.
  // This must be done on CPU for scalability (large problems violate max threads per block)
  checkCudaErrors(cudaMemcpy(d_nuzn,nuzn,sizeof(double)*nkt,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_nu1,nu1,sizeof(double)*nkt,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_kx,kx,sizeof(double)*nkt,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ky,ky,sizeof(double)*nkt,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_LL,LC,sizeof(int)*nkt,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ikF,ikF,sizeof(int)*n2,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ikN,ikN,sizeof(int)*n2,cudaMemcpyHostToDevice));
  //Set FFT Plans (Batched Z2D fails on large problems, D2Z is ok. At least up to 4096^2)
  (cufftPlan2d(&PlanZ2D,*NY,*NX,CUFFT_Z2D));
  (cufftPlan2d(&PlanD2Z,*NY,*NX,CUFFT_D2Z));
  (cufftPlanMany(&PlanBatchD2Z,2,np,NULL,1,0,NULL,1,0,CUFFT_D2Z,2));
  
  // Do a check of global memory use
  avail =0;
  total = 0;
  cudaMemGetInfo(&avail,&total);
  
  printf("\n total : %f MB \n",double(total)/(1024.0f*1024.0f));
  printf("\n avail : %f MB \n",double(avail)/(1024.0f*1024.0f));
  printf("\n used : %f MB \n",double(total-avail)/(1024.0f*1024.0f));

  //  RK_FFT(d_Z,d_ZR,PlanD2Z,d_ikN);

  if(*RecFLAG==1){
    // create index array for taking norms
    for(int iy=0; iy<2*normWidth; iy++){
      for(int ix=0; ix<nWidth2; ix++){
	int ik = (*KTY-normWidth+iy)*(*IKTX)+ix;
	iNorm[iy*nWidth2+ix]=ik;
      }
    }
    
    UPOZK = fopen("UPO_Zk.out","wb");
    UPOfile = fopen("UPOinfo_ts.dat","w");
    //        Write header for UPO Guesses output file                                                       
    fprintf(UPOfile, "Guess No : Start time : guess Period : guess Shift x: guess Shift y: guess Residual \n");
    for(int iShiftY=0; iShiftY < NSHIFTY; iShiftY++){
      shiftY[iShiftY] = twopi*iShiftY/(NSHIFTY);
    }
    for(int iShiftX=0; iShiftX < NSHIFTX; iShiftX++){
      shiftX[iShiftX] = twopi*iShiftX/(NSHIFTX*(*alpha)); 
    }
  } 
  vort = fopen("vort.dat","ab");
  if(*RecFLAG ==1 || *statsFLAG ==1){
    stats = fopen("stats.dat","w");
    points = fopen("points.dat","w");
    avgs = fopen("avgs.dat","wb");
    Zstore=(double2*)malloc(sizeof(double2)*nkt*storeSize);
    ZKtavg=(double2*)malloc(sizeof(double2)*nkt);
    checkCudaErrors(cudaMalloc((void**)&d_Urms,sizeof(cufftDoubleReal)*nr));
    checkCudaErrors(cudaMalloc((void**)&d_Vrms,sizeof(cufftDoubleReal)*nr));

    zeroReal<<<nblocks,nthreads>>>(d_Urms);
    zeroReal<<<nblocks,nthreads>>>(d_Vrms);

    for(int i=0; i<nkt; i++){
      ZKtavg[i].x=0.0;
      ZKtavg[i].y=0.0;
    }

    iStart = 0;
    nCopy = 0;
    checkCudaErrors(cudaMemcpy(Zstore,d_Z,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyDeviceToHost));
    normZ = computeNorm(Zstore,0);
    normZStore[0]=normZ;
  }

  zeroComplex<<<nblocks,nthreads>>>(d_NZK);

  // Do an initial set of velocity coeffs. Subsequently this occurs at the end of stepping kernels
  setVelocity<<<nblocks,nthreads>>>(d_Z, d_UK, d_VK,d_kx,d_ky,d_LL);

  // **************************
  //STEPPING STARTS HERE
  // **************************
  
  for(int NT=0; NT<*NSTOP; NT++){

    KR_FFT_ALL(d_Z,d_UK,d_VK,d_ZR,d_UR,d_VR,PlanZ2D,d_ikF);
    if((NT % (nOut)) == 0 && *statsFLAG==1 && *Time > 500.0) avgVelocity<<<nblocks,nthreads>>>(d_UR,d_VR,d_Urms,d_Vrms,d_LL,tavgCount);

    if(NT % nVort==0){
      checkCudaErrors(cudaMemcpy(ZR,(double *) d_ZR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
      fwrite(Time,sizeof(double),1,vort);
      fwrite(ZR,sizeof(double),nr,vort);
      fflush(vort);
    }
    multReal<<<nblocks,nthreads>>>(d_ZR,d_UR,d_VR,d_NZR);    // Do real space convolution, both terms inside NZR
	 
    RK_FFT(d_NZK,d_NZR,PlanBatchD2Z,d_ikN);    // RK does a batch of 2 ffts, previously NZK and UK
	  
    // Predictor step: prestep now does the end of 'convol', the step and resets velocity coeffs
    preStep<<<nblocks,nthreads>>>(d_Z,d_UK,d_VK,d_NZK,d_RHZ,d_nuzn,d_nu1,d_kx,d_ky,d_LL);
	 
    KR_FFT_ALL(d_Z,d_UK,d_VK,d_ZR,d_UR,d_VR,PlanZ2D,d_ikF);    // Second half of time step:
    multReal<<<nblocks,nthreads>>>(d_ZR,d_UR,d_VR,d_NZR);
    RK_FFT(d_NZK,d_NZR,PlanBatchD2Z,d_ikN);

    //Correction step: corStep is analagous in structure to prestep.
    corStep<<<nblocks,nthreads>>>(d_Z,d_UK,d_VK,d_NZK,d_RHZ,d_nu1,d_kx,d_ky,d_LL);
    if((NT % (nOut)) == 0 && (*RecFLAG ==1 || *statsFLAG==1) && NT!=0){
      
      if(nStore >= 1 && *RecFLAG==1){
	recurrence_check(Zstore,kx,ky,NT,nCopy); 
	fflush(stdout);
     }
      //copy Z into ZStore

      // keep track of indexing for the copy index and the back search start index
      // copy index should lag the start index
      nStore++;
      if(nStore >= storeSize){
	nStore = storeSize;
	nCopy++;
	nCopy -= floor(nCopy/storeSize)*storeSize;
	iStart=nCopy;
      }else{
	nCopy = nStore;
      }

      checkCudaErrors(cudaMemcpy(Zstore+nCopy*nkt,d_Z,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyDeviceToHost));
      normZStore[nCopy]=computeNorm(Zstore,nCopy);
      if(*statsFLAG==1)computeStats(Zstore,ZKtavg,kx,ky,LC,nCopy);
    }
    *Time = *TSTART + (NT+1)*(*DELT);
    tme = *Time;// Increase time
    fflush(stdout);

  }

  //Perform final check:
  if(*RecFLAG ==1){
    recurrence_check(Zstore,kx,ky,*NSTOP,nCopy);
    fclose(UPOZK);
    fclose(UPOfile);
  }

  // Copy final state off GPU

  checkCudaErrors(cudaMemcpy(Z, d_Z, sizeof(cufftDoubleComplex)*nkt, cudaMemcpyDeviceToHost));

  //  subLam<<<1,1>>>(d_Z,v2);
  //KR_FFT_ALL(d_Z,d_UK,d_VK,d_ZR,d_UR,d_VR,PlanZ2D,d_ikF);
  checkCudaErrors(cudaMemcpy(ZR,(double *) d_ZR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
  fwrite(Time,sizeof(double),1,vort);
  fwrite(ZR,sizeof(double),nr,vort);
  fflush(vort);

  if(*statsFLAG==1){
    checkCudaErrors(cudaMemcpy(d_Z,ZKtavg,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyHostToDevice));  
    setVelocity<<<nblocks,nthreads>>>(d_Z, d_UK, d_VK,d_kx,d_ky,d_LL);
    KR_FFT_ALL(d_Z,d_UK,d_VK,d_ZR,d_UR,d_VR,PlanZ2D,d_ikF);
    
    checkCudaErrors(cudaMemcpy(ZR,(double *) d_UR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(ZR,sizeof(double),nr,avgs);
    
    checkCudaErrors(cudaMemcpy(ZR,(double *) d_VR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(ZR,sizeof(double),nr,avgs);
    
    subtractReal<<<nblocks,nthreads>>>(d_UR,d_Urms);
    
    checkCudaErrors(cudaMemcpy(ZR,(double *) d_Vrms, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(ZR,sizeof(double),nr,avgs);
    
    checkCudaErrors(cudaMemcpy(ZR,(double *) d_UR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(ZR,sizeof(double),nr,avgs);
    
    fflush(avgs);
    
    fclose(avgs);
    fclose(stats);
    fclose(points);

    checkCudaErrors(cudaFree(d_Urms));
    checkCudaErrors(cudaFree(d_Vrms));
    free(Zstore);
    free(ZKtavg);
  }


    fclose(vort);
  // Free GPU global memory
  checkCudaErrors(cudaFree(d_Z));
  checkCudaErrors(cudaFree(d_UK));
  checkCudaErrors(cudaFree(d_VK));
  checkCudaErrors(cudaFree(d_NZK));
  checkCudaErrors(cudaFree(d_RHZ));
  
  checkCudaErrors(cudaFree(d_ZR));	
  checkCudaErrors(cudaFree(d_UR));
  checkCudaErrors(cudaFree(d_VR));
  checkCudaErrors(cudaFree(d_NZR));
  
  checkCudaErrors(cudaFree(d_nuzn));
  checkCudaErrors(cudaFree(d_nu1));
  checkCudaErrors(cudaFree(d_ikF));
  checkCudaErrors(cudaFree(d_ikN));
  checkCudaErrors(cudaFree(d_kx));
  checkCudaErrors(cudaFree(d_ky));
  checkCudaErrors(cudaFree(d_LL));
  
  //Destroy fft plans
  (cufftDestroy(PlanZ2D));
  (cufftDestroy(PlanBatchD2Z));

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
