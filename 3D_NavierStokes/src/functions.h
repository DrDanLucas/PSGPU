///////////////////////////////////////////////////////////////////////////////

extern "C" void set_gpu_(int *go)
{
  int            dev, deviceCount;
  cudaDeviceProp devProp;

  (cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("cutil error: no devices supporting CUDA\n");
    exit(-1);
  }

  (cudaGetDevice(&dev));
  //  (cudaSetDevice(*go));
  (cudaGetDeviceProperties(&devProp,dev));
  printf("\n Using CUDA device %d: %s\n\n", dev,devProp.name);


}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
////////////////////////

extern "C" void KR_FFT_ALL(cufftDoubleComplex *d_xii,
			   cufftDoubleComplex *d_eta,
			   cufftDoubleComplex *d_zet, 
			   cufftDoubleComplex *d_UK,
			   cufftDoubleComplex *d_VK,
			   cufftDoubleComplex *d_WK, 
			   cufftDoubleReal*d_xir, 
			   cufftDoubleReal*d_etr, 
			   cufftDoubleReal*d_ztr, 
			   cufftDoubleReal*d_UR, 
			   cufftDoubleReal*d_VR,
			   cufftDoubleReal*d_WR,
			   cufftHandle PlanZ2D,
			   int *d_ikF, 
			   int n2){

  cufftDoubleComplex *d_FF,*d_tmp;

  //This version does each variable separately (FFT for each Z, U and V rather than one batched FFT)

  (cudaMalloc((void**)&d_FF,sizeof(cufftDoubleComplex)*(n2)));
  (cudaMalloc((void**)&d_tmp,sizeof(cufftDoubleComplex)*(nkt)));

  conj<<<nblocks,nthreads>>>(d_xii,d_tmp,d_kx);
  setFFN<<<nblocks,nthreads>>>(d_tmp,d_FF,d_ikF);  // Pad truncated wave numbers
  (cufftExecZ2D(PlanZ2D,d_FF,d_xir));  // Do Z FFT

  conj<<<nblocks,nthreads>>>(d_eta,d_tmp,d_kx);
  setFFN<<<nblocks,nthreads>>>(d_tmp,d_FF,d_ikF);  // Pad truncated wave numbers
  (cufftExecZ2D(PlanZ2D,d_FF,d_etr));  // Do Z FFT

  conj<<<nblocks,nthreads>>>(d_zet,d_tmp,d_kx);
  setFFN<<<nblocks,nthreads>>>(d_tmp,d_FF,d_ikF);  // Pad truncated wave numbers
  (cufftExecZ2D(PlanZ2D,d_FF,d_ztr));  // Do Z FFT

  conj<<<nblocks,nthreads>>>(d_VK,d_tmp,d_kx);
  setFFN<<<nblocks,nthreads>>>(d_tmp,d_FF,d_ikF);  // Pad truncated wave numbers
  (cufftExecZ2D(PlanZ2D,d_FF,d_VR));  // Do V FFT

  conj<<<nblocks,nthreads>>>(d_UK,d_tmp,d_kx);
  setFFN<<<nblocks,nthreads>>>(d_tmp,d_FF,d_ikF);  // Pad truncated wave numbers
  (cufftExecZ2D(PlanZ2D,d_FF,d_UR));  // Do U FFT

  conj<<<nblocks,nthreads>>>(d_WK,d_tmp,d_kx);
  setFFN<<<nblocks,nthreads>>>(d_tmp,d_FF,d_ikF);  // Pad truncated wave numbers
  (cufftExecZ2D(PlanZ2D,d_FF,d_WR));  // Do V FFT

  (cudaFree(d_FF));
  (cudaFree(d_tmp));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
////////////////////////

extern "C" void RK_FFT(cufftDoubleComplex *d_NXK,
		       cufftDoubleComplex *d_NYK,
		       cufftDoubleComplex *d_NZK, 
		       cufftDoubleReal*d_XR,
		       cufftDoubleReal*d_YR,
		       cufftDoubleReal*d_ZR,
		       cufftHandle PlanD2Z,
		       int *d_ikN,
		       int n2){
  
  cufftDoubleComplex *d_FF;

  //Doing a batch of 2 2D FFTs so that now NZK stores the spectral UK and NZK from the old convol in that order.

  (cudaMalloc((void**)&d_FF,sizeof(cufftDoubleComplex)*(n2)));

  (cufftExecD2Z(PlanD2Z,d_XR,d_FF));

  normFF1<<<nblocks,nthreads>>>(d_NXK,d_FF,d_ikN);  // normalise output

  (cufftExecD2Z(PlanD2Z,d_YR,d_FF));

  normFF1<<<nblocks,nthreads>>>(d_NYK,d_FF,d_ikN);  // normalise output

  (cufftExecD2Z(PlanD2Z,d_ZR,d_FF));

  normFF1<<<nblocks,nthreads>>>(d_NZK,d_FF,d_ikN);  // normalise output

  (cudaFree(d_FF));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" double computeNorm(double2 *XIIstore,double2 *ETAstore,double2 *ZETstore,int *LL,int iStore){

  double normX=0.0;
  double normE=0.0;
  double normZ=0.0;
  for(int i=0; i<nNorm; i++){
    int ik = iNorm[i]+iStore*nkt;
    if(LL[iNorm[i]] == 1){
	normX += XIIstore[ik].x*XIIstore[ik].x + XIIstore[ik].y*XIIstore[ik].y;
	normE += ETAstore[ik].x*ETAstore[ik].x + ETAstore[ik].y*ETAstore[ik].y;
	normZ += ZETstore[ik].x*ZETstore[ik].x + ZETstore[ik].y*ZETstore[ik].y;
      }
  }
  normZ += normX+normE;
  return normZ;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void computeStats(double2 *XIIstore,double2 *ETAstore,double2 *ZETstore,double2 *XIItavg,double2 *ETAtavg,double2 *ZETtavg,double *kx, double *ky,double *kz,int *LL,int iStore){

  double eturb = 0.0;
  double dturb = 0.0;
  double XIa = 0;
  double ETa = 0;
  double ZTa = 0;
  double Ua = 0;
  double Va = 0;
  double Wa = 0;

  energy = 0.0;
  diss = 0.0;
  input = 0.0;
  for(int i=0; i<nkt; i++){
    if(LL[i] == 1){
    int ik = i + iStore*nkt;
    double wk = max(kx[i]*kx[i] + ky[i]*ky[i]+kz[i]*kz[i],0.001);

    double VZX = XIIstore[ik].x*XIIstore[ik].x + XIIstore[ik].y*XIIstore[ik].y;
    double VZY = ETAstore[ik].x*ETAstore[ik].x + ETAstore[ik].y*ETAstore[ik].y;
    double VZZ = ZETstore[ik].x*ZETstore[ik].x + ZETstore[ik].y*ZETstore[ik].y;
    double VZ = VZX+VZY+VZZ;
    diss += VZ;
    energy += VZ/wk;

    XIa+=VZX;
    ETa+=VZY;
    ZTa+=VZZ;
    Ua += pow((ky[i]*ZETstore[ik].y-kz[i]*ETAstore[ik].y)/wk,2) + pow((kz[i]*ETAstore[ik].x-ky[i]*ZETstore[ik].x)/wk,2);
    Va += pow((kz[i]*XIIstore[ik].y-kx[i]*ZETstore[ik].y)/wk,2) + pow((kx[i]*ZETstore[ik].x-kz[i]*XIIstore[ik].x)/wk,2);
    Wa += pow((kx[i]*ETAstore[ik].y-ky[i]*XIIstore[ik].y)/wk,2) + pow((ky[i]*XIIstore[ik].x-kx[i]*ETAstore[ik].x)/wk,2);

    XIItavg[i].x = (XIItavg[i].x*tavgCount + XIIstore[ik].x)/(tavgCount+1.0);
    XIItavg[i].y = (XIItavg[i].y*tavgCount + XIIstore[ik].y)/(tavgCount+1.0);
    ETAtavg[i].x = (ETAtavg[i].x*tavgCount + ETAstore[ik].x)/(tavgCount+1.0);
    ETAtavg[i].y = (ETAtavg[i].y*tavgCount + ETAstore[ik].y)/(tavgCount+1.0);
    ZETtavg[i].x = (ZETtavg[i].x*tavgCount + ZETstore[ik].x)/(tavgCount+1.0);
    ZETtavg[i].y = (ZETtavg[i].y*tavgCount + ZETstore[ik].y)/(tavgCount+1.0);

    VZX = XIItavg[i].x*XIItavg[i].x + XIItavg[i].y*XIItavg[i].y;
    VZY = ETAtavg[i].x*ETAtavg[i].x + ETAtavg[i].y*ETAtavg[i].y;
    VZZ = ZETtavg[i].x*ZETtavg[i].x + ZETtavg[i].y*ZETtavg[i].y;
    VZ = VZX+VZY+VZZ;

    eturb += VZ/wk;
    dturb += VZ;
    if(kx[i] ==0.0 && ky[i] == kfy && kz[i] == kfy){
      input +=  ampfor*XIIstore[ik].x;
    }
    if(kx[i] ==0.0 && ky[i] == kfy && kz[i] == -kfy){
      input -=  ampfor*XIIstore[ik].x;
      }
    }
  }
  input *= v2*kfy*kfy;
  diss *= v2*v2*8.0*kfy*kfy;
  dturb *= v2*16.0*kfy*kfy;
  //now all stats normalised with laminar

  dissMax = max(diss,dissMax);
  dissMin = min(diss,dissMin);
  enerMax = max(energy,enerMax);
  enerMin = min(energy,enerMin);
  inputMax = max(input,inputMax);
  inputMin = min(input,inputMin);

  dissAvg = (dissAvg*tavgCount + diss)/(tavgCount+1.0);
  enerAvg = (enerAvg*tavgCount + energy)/(tavgCount+1.0);
  inputAvg = (inputAvg*tavgCount + input)/(tavgCount+1.0);

  int istat= int(NT/nOut)-1;
  Diss[istat]  = diss;
  Energy[istat]= energy;
  Input[istat] = input;

  tavgCount++;
  fprintf(stats,"%e %e %e %e %e %e \n",tme,energy,eturb,diss,dturb,input);
  fflush(stats);
  fprintf(engy,"%e %e %e %e %e %e %e \n",tme,Ua,Va,Wa,XIa,ETa,ZTa);
  fflush(engy);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void computeZdash(double2 *XIIstore,
			     double2 *ETAstore,
			     double2 *ZETstore,
			     double2 *XIIdash,
			     double2 *ETAdash,
                             double2 *ZETdash,
                             double *kx,
                             double *ky,
                             double *kz,
                             int iStore
                             ){

  for(int iShiftY=0; iShiftY < NSHIFTY; iShiftY++){
    for(int iShiftZ=0; iShiftZ < NSHIFTZ; iShiftZ++){
      for(int iShiftX=0; iShiftX < NSHIFTX; iShiftX++){
      
	for(int i=0; i<nNorm; i++){
	  int ik =iNorm[i];
	  int ikk =ik+iStore*nkt;
	  int iks = i + nNorm*(iShiftX+NSHIFTX*(iShiftY+NSHIFTY*iShiftZ));

	  double kkx = kx[ik];
	  double kky = ky[ik];
	  double kkz = kz[ik];
	  double shift = kkx*shiftX[iShiftX]+kky*shiftY[iShiftY]+kkz*shiftZ[iShiftZ];
	  
	  XIIdash[iks].x = cos(shift)*XIIstore[ikk].x+sin(shift)*XIIstore[ikk].y;
	  XIIdash[iks].y = cos(shift)*XIIstore[ikk].y-sin(shift)*XIIstore[ikk].x;

	  ETAdash[iks].x = cos(shift)*ETAstore[ikk].x+sin(shift)*ETAstore[ikk].y;
	  ETAdash[iks].y = cos(shift)*ETAstore[ikk].y-sin(shift)*ETAstore[ikk].x;

	  ZETdash[iks].x = cos(shift)*ZETstore[ikk].x+sin(shift)*ZETstore[ikk].y;
	  ZETdash[iks].y = cos(shift)*ZETstore[ikk].y-sin(shift)*ZETstore[ikk].x;
	}
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void recurrence_check(double2 *XIIstore,
				 double2 *ETAstore,
				 double2 *ZETstore,
                                 double *kx,
                                 double *ky,
                                 double *kz,
                                 int NT,
                                 int nCopy
                                 ){

  double minSx,minSy,minSz,resPrev;
  double minRESIDUALinner = 10000.0;
  double minPERIOD = 10000.0;
  double res = 0.0;
  double normFN = 0.0;
  double normFNN = 0.0;
  int minPflag =1;
  int minSTOREinner =0;
  int minCOUNTinner =0;
  double2 *XIIdash,*ETAdash,*ZETdash;

  XIIdash = (double2*)malloc(sizeof(double2)*nNorm*NSHIFTX*NSHIFTY*NSHIFTZ);
  ETAdash = (double2*)malloc(sizeof(double2)*nNorm*NSHIFTX*NSHIFTY*NSHIFTZ);
  ZETdash = (double2*)malloc(sizeof(double2)*nNorm*NSHIFTX*NSHIFTY*NSHIFTZ);
  
  computeZdash(XIIstore,ETAstore,ZETstore,XIIdash,ETAdash,ZETdash,kx,ky,kz,nCopy);

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
      for(int iShiftZ=0; iShiftZ < NSHIFTZ; iShiftZ++){
	for(int iShiftX=0; iShiftX < NSHIFTX; iShiftX++){
	  int is = nNorm*(iShiftX+NSHIFTX*(iShiftY+NSHIFTY*iShiftZ));
	  double nZdash = 0.0;
	  normFNN = 0.0;
	  for(int i=0; i<nNorm;i++){
	    int ik = iNorm[i]+iStore*nkt;

	    normFNN -= 2.0*(XIIdash[is+i].x*XIIstore[ik].x + XIIdash[is+i].y*XIIstore[ik].y);
	    nZdash += XIIdash[is+i].x*XIIdash[is+i].x + XIIdash[is+i].y*XIIdash[is+i].y;

	    normFNN -= 2.0*(ETAdash[is+i].x*ETAstore[ik].x + ETAdash[is+i].y*ETAstore[ik].y);
	    nZdash += ETAdash[is+i].x*ETAdash[is+i].x + ETAdash[is+i].y*ETAdash[is+i].y;

	    normFNN -= 2.0*(ZETdash[is+i].x*ZETstore[ik].x + ZETdash[is+i].y*ZETstore[ik].y);
	    nZdash += ZETdash[is+i].x*ZETdash[is+i].x + ZETdash[is+i].y*ZETdash[is+i].y;
	  }
        normFN = normFNN + nZdash + normZStore[iStore];

        if(sqrt(normFN/normZStore[nCopy]) < res){
          res = sqrt(normFN/normZStore[nCopy]);
          minSx = shiftX[iShiftX];
          minSy = shiftY[iShiftY];
          minSz = shiftZ[iShiftZ];
        }
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
      minSHIFTZinner = minSz;
    }

  }
	
  // When min Residual first drops below threshold start new output sequence
  if( minRESIDUALinner < ResidualThreshold && RecOUTflag == 0 ){ 
    printf("------------------------------------------------\n");
    printf("Time = %e UPO Guess found \n",tme);
    minRESIDUALouter = ResidualThreshold;
    RecOUTflag = 1;
  }

  // Locate minimum residual for current output sequence and store corresponding data
  if( minRESIDUALinner < minRESIDUALouter && RecOUTflag==1 ){
    minRESIDUALouter = minRESIDUALinner;
    minSTOREouter =  minSTOREinner;
    minTIMEouter = tme + minCOUNTinner*nOut*delt;
    minSHIFTXouter = minSHIFTXinner;
    minSHIFTYouter = minSHIFTYinner;
    minSHIFTZouter = minSHIFTZinner;
    RecPERIOD = ((minCOUNTinner)*nOut)*(delt);
  }

  // Output best UPO guess at end of current output sequence
  if( minRESIDUALinner > ResidualThreshold && RecOUTflag==1 ){
    RecOUTflag = 0;

    fwrite(&minTIMEouter,sizeof(double),1,UPOZK);
    fwrite(&v2,sizeof(double),1,UPOZK);
    fwrite(&RecPERIOD,sizeof(double),1,UPOZK);
    fwrite(&minSHIFTXouter,sizeof(double),1,UPOZK);
    fwrite(&minSHIFTYouter,sizeof(double),1,UPOZK);
    fwrite(&minSHIFTZouter,sizeof(double),1,UPOZK);
    fwrite(&minRESIDUALouter,sizeof(double),1,UPOZK);
    fwrite(XIIstore+minSTOREouter*nkt,sizeof(double2),nkt,UPOZK);
    fwrite(ETAstore+minSTOREouter*nkt,sizeof(double2),nkt,UPOZK);
    fwrite(ZETstore+minSTOREouter*nkt,sizeof(double2),nkt,UPOZK);
    izkout = izkout +1;
    printf("Wrote to UPO_Zk.out number %d at time %e \n",izkout,tme);

    fprintf(UPOfile,"%d %e %e %e %e %e %e \n",izkout, minTIMEouter,RecPERIOD,minSHIFTXouter,minSHIFTYouter,minSHIFTZouter,minRESIDUALouter);
  }

  free(XIIdash);
  free(ETAdash);
  free(ZETdash);
  fflush(UPOfile);
  fflush(UPOZK);
}


