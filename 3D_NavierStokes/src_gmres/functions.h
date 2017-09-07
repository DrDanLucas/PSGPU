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

  // cudaGetDevice(&dev));
  (cudaSetDevice(*go));
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



