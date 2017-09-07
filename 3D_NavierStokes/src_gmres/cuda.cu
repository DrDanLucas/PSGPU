#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "functions.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
////////////////////////

extern "C" void timestep_cuda_(double2* xii,
			       double2* eta,
			       double2* zet,
			       double* nuzn,
			       double* nu1,
			       double* kx,
			       double* ky,
			       double* kz,
			       double* Time, 
			       double* TSTART,
			       double* AMPFOR,
			       double* DELT,
			       int* KFY,
			       double* V2,
			       int* ikF,
			       int* ikN,
			       int* LC,
			       int* NSTOP
			       )
{
  
  // Define CPU variables
  n2 = nx2*ny*nz;
  in = 1.0/nr;
  tme = 0.0;
  tstart = *TSTART;
  delt = *DELT;  
  ampfor = *AMPFOR;
  v2 = *V2;
  kfy = double(*KFY);

  // Define global device variables
  int *d_ikF, *d_ikN, *d_LL;
  cufftDoubleComplex *d_xii,*d_eta,*d_zet, *d_UK, *d_VK, *d_WK, *d_RHX,*d_RHY,*d_RHZ;
  cufftDoubleReal *d_xir,*d_etr,*d_ztr, *d_UR,*d_VR,*d_WR;
  double *d_nuzn,*d_nu1;
  cufftHandle PlanZ2D,PlanD2Z;
  
  // Allocate global memory on GPU. (Constant memory does not need allocating) 	
  (cudaMalloc((void**)&d_xii,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_eta,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_zet,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_UK,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_VK,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_WK,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_RHX,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_RHY,sizeof(cufftDoubleComplex)*(nkt)));
  (cudaMalloc((void**)&d_RHZ,sizeof(cufftDoubleComplex)*(nkt)));
  
  (cudaMalloc((void**)&d_xir,sizeof(cufftDoubleReal)*(nr)));
  (cudaMalloc((void**)&d_etr,sizeof(cufftDoubleReal)*(nr)));
  (cudaMalloc((void**)&d_ztr,sizeof(cufftDoubleReal)*(nr)));
  (cudaMalloc((void**)&d_UR,sizeof(cufftDoubleReal)*nr));
  (cudaMalloc((void**)&d_VR,sizeof(cufftDoubleReal)*nr));
  (cudaMalloc((void**)&d_WR,sizeof(cufftDoubleReal)*nr));
  
  (cudaMalloc((void**)&d_nuzn,sizeof(double)*(nkt)));
  (cudaMalloc((void**)&d_nu1,sizeof(double)*(nkt)));
  (cudaMalloc((void**)&d_kx,sizeof(double)*(nkt)));
  (cudaMalloc((void**)&d_ky,sizeof(double)*(nkt)));
  (cudaMalloc((void**)&d_kz,sizeof(double)*(nkt)));
  (cudaMalloc((void**)&d_LL,sizeof(int)*nkt));
  (cudaMalloc((void**)&d_ikN,sizeof(int)*nkt));
  (cudaMalloc((void**)&d_ikF,sizeof(int)*n2));
  
  // Copy state data to GPU global memory 
  (cudaMemcpy(d_xii,xii,sizeof(cufftDoubleComplex)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_eta,eta,sizeof(cufftDoubleComplex)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_zet,zet,sizeof(cufftDoubleComplex)*(nkt),cudaMemcpyHostToDevice));
  //Set up various arrays to enable generic kernel calls
  //i.e. calculate indexing for padding either side of FFTs, wavenumber arrays, mask, and timestep arrays.
  // This must be done on CPU for scalability (large problems violate max threads per block)

  (cudaMemcpy(d_nuzn,nuzn,sizeof(double)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_nu1,nu1,sizeof(double)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_kx,kx,sizeof(double)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_ky,ky,sizeof(double)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_kz,kz,sizeof(double)*(nkt),cudaMemcpyHostToDevice));
  (cudaMemcpy(d_ikF,ikF, sizeof(int)*n2,cudaMemcpyHostToDevice));
  (cudaMemcpy(d_ikN,ikN,sizeof(int)*nkt,cudaMemcpyHostToDevice));
  (cudaMemcpy(d_LL,LC,sizeof(int)*nkt,cudaMemcpyHostToDevice));

  // Copy constsant parameters to GPU constant memory
  (cudaMemcpyToSymbol(d_IN,&in,sizeof(double)));
  (cudaMemcpyToSymbol(d_AMPFOR,AMPFOR,sizeof(double)));
  (cudaMemcpyToSymbol(d_DELT,DELT,sizeof(double)));
  (cudaMemcpyToSymbol(d_v2,V2,sizeof(double)));
  (cudaMemcpyToSymbol(d_KFY,&kfy,sizeof(double)));
  (cudaMemcpyToSymbol(d_O2,&n2,sizeof(int)));
    
  //Set FFT Plans
  (cufftPlan3d(&PlanZ2D,nz,ny,nx,CUFFT_Z2D));
  (cufftPlan3d(&PlanD2Z,nz,ny,nx,CUFFT_D2Z));

  setVelocity<<<nblocks,nthreads>>>(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_kx,d_ky,d_kz,d_LL);
 
  // **************************
  //STEPPING STARTS HERE
  // **************************
  fflush(stdout);
  
  for(int NT=0; NT<*NSTOP; NT++){
    
    KR_FFT_ALL(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_xir,d_etr,d_ztr,d_UR,d_VR,d_WR,PlanZ2D,d_ikF,n2);

    multReal<<<nblocks,nthreads>>>(d_xir,d_etr,d_ztr,d_UR,d_VR,d_WR); // real space convolution, result in U,V,W
	 
    RK_FFT(d_UK,d_VK,d_WK,d_UR,d_VR,d_WR,PlanD2Z,d_ikN,n2);
	  
    // Predictor step: prestep now does the end of 'convol', the step and resets velocity coeffs
    preStep<<<nblocks,nthreads>>>(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_RHX,d_RHY,d_RHZ,d_nuzn,d_nu1,d_kx,d_ky,d_kz,d_LL);
    
    KR_FFT_ALL(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_xir,d_etr,d_ztr,d_UR,d_VR,d_WR,PlanZ2D,d_ikF,n2);

    multReal<<<nblocks,nthreads>>>(d_xir,d_etr,d_ztr,d_UR,d_VR,d_WR); // real space convolution, result in U,V,W
	 
    RK_FFT(d_UK,d_VK,d_WK,d_UR,d_VR,d_WR,PlanD2Z,d_ikN,n2);

    *Time = *TSTART + (NT+1)*(*DELT);    // Increase time
    tme += *DELT;

    //Correction step: corStep is analagous in structure to prestep.
    corStep<<<nblocks,nthreads>>>(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_RHX,d_RHY,d_RHZ,d_nuzn,d_nu1,d_kx,d_ky,d_kz,d_LL);

  }
  fflush(stdout);

    // Copy state off GPU

  (cudaMemcpy(xii, d_xii, sizeof(cufftDoubleComplex)*(nkt), cudaMemcpyDeviceToHost));
  (cudaMemcpy(eta, d_eta, sizeof(cufftDoubleComplex)*(nkt), cudaMemcpyDeviceToHost));
  (cudaMemcpy(zet, d_zet, sizeof(cufftDoubleComplex)*(nkt), cudaMemcpyDeviceToHost));

  // Free GPU global memory
  (cudaFree(d_xii));
  (cudaFree(d_eta));
  (cudaFree(d_zet));
  (cudaFree(d_UK));
  (cudaFree(d_VK));
  (cudaFree(d_WK));
  (cudaFree(d_RHX));
  (cudaFree(d_RHY));
  (cudaFree(d_RHZ));
  
  (cudaFree(d_xir));	
  (cudaFree(d_etr));	
  (cudaFree(d_ztr));	
  (cudaFree(d_UR));
  (cudaFree(d_VR));
  (cudaFree(d_WR));
  
  (cudaFree(d_nuzn));
  (cudaFree(d_nu1));
  (cudaFree(d_LL));
  (cudaFree(d_ikF));
  (cudaFree(d_ikN));
  (cudaFree(d_kx));
  (cudaFree(d_ky));
  (cudaFree(d_kz));

  //Destroy fft plans
  (cufftDestroy(PlanZ2D));
  (cufftDestroy(PlanD2Z));

  fflush(stdout);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
