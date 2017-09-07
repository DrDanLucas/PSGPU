  fflush(stdout);

    // Copy state off GPU
  (cudaMemcpy(xii, d_xii, sizeof(cufftDoubleComplex)*nkt, cudaMemcpyDeviceToHost));
  (cudaMemcpy(eta, d_eta, sizeof(cufftDoubleComplex)*nkt, cudaMemcpyDeviceToHost));
  (cudaMemcpy(zet, d_zet, sizeof(cufftDoubleComplex)*nkt, cudaMemcpyDeviceToHost));
    
  // Do a final KR call if we want to visualise physical data (include a flag here later)
  (cudaMemcpy(xir, d_xir, sizeof(double)*nr, cudaMemcpyDeviceToHost));
  (cudaMemcpy(etr, d_etr, sizeof(double)*nr, cudaMemcpyDeviceToHost));
  (cudaMemcpy(ztr, d_ztr, sizeof(double)*nr, cudaMemcpyDeviceToHost)); 

  Umean=(double*)malloc(sizeof(double)*ny*nz);
  Vmean=(double*)malloc(sizeof(double)*ny*nz);
  Wmean=(double*)malloc(sizeof(double)*ny*nz);
  uhat=(double*)malloc(sizeof(double)*ny*nz);
  vhat=(double*)malloc(sizeof(double)*ny*nz);
  what=(double*)malloc(sizeof(double)*ny*nz);
if(*statsFLAG==1 && *NSTOP > 1){ // do last statistics and compute/output some mean profiles
    NT++;
    computeStats(xii,eta,zet,XIItavg,ETAtavg,ZETtavg,kx,ky,kz,LC,0);      

    (cudaMemcpy(d_xii,XIItavg,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyHostToDevice));
    (cudaMemcpy(d_eta,ETAtavg,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyHostToDevice));
    (cudaMemcpy(d_zet,ZETtavg,sizeof(cufftDoubleComplex)*nkt,cudaMemcpyHostToDevice));
    setVelocity<<<nblocks,nthreads>>>(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_kx,d_ky,d_kz,d_LL);
    KR_FFT_ALL(d_xii,d_eta,d_zet,d_UK,d_VK,d_WK,d_xir,d_etr,d_ztr,d_UR,d_VR,d_WR,PlanZ2D,d_ikF,n2);

    (cudaMemcpy(xir,(double *) d_UR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(xir,sizeof(double),nr,avgs);

    for(int iz=0; iz<nz; iz++){
      for(int iy=0; iy<ny; iy++){
    	int ik = iz*ny+iy;
    	Umean[ik] = 0.0;
	
    	for(int ix=0; ix<nx; ix++){
    	  int ikt = (ny*iz+iy)*nx+ix;
    	  Umean[ik] += xir[ikt];
    	}
    	Umean[ik] /= double(nx);
      }
    }

    (cudaMemcpy(etr,(double *) d_VR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(etr,sizeof(double),nr,avgs);

    for(int iz=0; iz<nz; iz++){
      for(int iy=0; iy<ny; iy++){
    	int ik = iz*ny+iy;
    	Vmean[ik] = 0.0;
	
    	for(int ix=0; ix<nx; ix++){
    	  int ikt = (ny*iz+iy)*nx+ix;
    	  Vmean[ik] += etr[ikt];
    	}
    	Vmean[ik] /= double(nx);
      }
    }

    (cudaMemcpy(ztr,(double *) d_WR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(ztr,sizeof(double),nr,avgs);

    for(int iz=0; iz<nz; iz++){
      for(int iy=0; iy<ny; iy++){
        int ik = iz*ny+iy;
        Wmean[ik] = 0.0;

        for(int ix=0; ix<nx; ix++){
          int ikt = (ny*iz+iy)*nx+ix;
          Wmean[ik] += ztr[ikt];
        }
        Wmean[ik] /= double(nx);
      }
    }

    subtractReal<<<nblocks,nthreads>>>(d_UR,d_Urms);

    (cudaMemcpy(etr,(double *) d_Vrms, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(etr,sizeof(double),nr,avgs);
    for(int iz=0; iz<nz; iz++){
      for(int iy=0; iy<ny; iy++){
        int ik = iz*ny+iy;
        vhat[ik] = 0.0;

        for(int ix=0; ix<nx; ix++){
          int ikt = (ny*iz+iy)*nx+ix;
          vhat[ik] += etr[ikt];
        }
        vhat[ik] /= double(nx);
      }
    }

    (cudaMemcpy(ztr,(double *) d_UR, sizeof(double)*nr, cudaMemcpyDeviceToHost));
    fwrite(Time,sizeof(double),1,avgs);
    fwrite(ztr,sizeof(double),nr,avgs);

    for(int iz=0; iz<nz; iz++){
      for(int iy=0; iy<ny; iy++){
        int ik = iz*ny+iy;
        uhat[ik] = 0.0;

        for(int ix=0; ix<nx; ix++){
          int ikt = (ny*iz+iy)*nx+ix;
          uhat[ik] += ztr[ikt];
        }
        uhat[ik] /= double(nx);
    	fprintf(means,"%.15e %.15e %.15e %.15e %.15e %.15e \n",iy*twopi/double(ny),iz*twopi/double(nz),Umean[ik],Vmean[ik],uhat[ik],vhat[ik]);
      }
    }

  printf("AVERAGE STATS *****\n");
  printf("energy = %e,  eturb = %e \n",enerAvg,eturbAvg);
  printf("dissipation = %e,  dturb = %e \n",dissAvg,dturbAvg);
  printf("input E = %e \n",inputAvg);

  printf("max/min dissipation %e, %e \n ", dissMax,dissMin);
  printf("max/min energy %e, %e \n", enerMax,enerMin);
  printf("max/min input %e, %e \n", inputMax,inputMin);


  // Compute the p.d.f.s of dissipation/energy/input
  if(dissMax-dissMin > 0.001){
    int nBin = 100;
    double correct = 1.0000001;
    int I_E[nBin],I_D[nBin],I_I[nBin];

    double dD = (dissMax*correct-dissMin)/double(nBin);
    double dI = (inputMax*correct-inputMin)/double(nBin);
    double dE = (enerMax*correct-enerMin)/double(nBin);

    for(int ib=0; ib<nBin; ib++){
      I_E[ib] = 0;
      I_D[ib] = 0;
      I_I[ib] = 0;
    }
    printf("dD = %e \n",dD);
    printf("dI = %e \n",dI);
    printf("dE = %e \n",dE);
    int j=0;
    for(NT = 0; NT < nStats-1; NT++){      
      j=int((Energy[NT]-enerMin)/dE) +1;
      printf("j= %d \n",j);
      I_E[j]++;
      j=int((Diss[NT]-dissMin)/dD) +1;
      if(j>nBin) printf("error in j!");
      I_D[j]++;
      j=int((Input[NT]-inputMin)/dI) +1;
      if(j>nBin) printf("error in j!");
      I_I[j]++;
    }
    printf("dD = %e \n",dD);
    printf("dI = %e \n",dI);
    printf("dE = %e \n",dE);
    fflush(stdout);

    for(int ib=0; ib<nBin; ib++){
      fprintf(pdfs,"%.15e %.15e %.15e %.15e %.15e %.15e \n", enerMin+(ib-0.5)*dE , I_E[ib]/((nStats-1)*dE)
  	      , dissMin+(ib-0.5)*dD , I_D[ib]/((nStats-1)*dD)
  	      ,inputMin+(ib-0.5)*dI , I_I[ib]/((nStats-1)*dI));
    }
  }
  }
  free(Umean);
  free(Vmean);
  free(Wmean);
  free(uhat);
  free(vhat);
  free(what);

  printf("pdfs and averages done \n"); 
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
  
  (cudaFree(d_LL));
  (cudaFree(d_ikF));
  (cudaFree(d_ikN));
  (cudaFree(d_kx));
  (cudaFree(d_ky));
  (cudaFree(d_kz));
  (cudaFree(d_nuzn));
  (cudaFree(d_nu1));
  printf("GPU free done \n"); 
  //Free CPU memory
  if(*statsFLAG !=0){
    free(XIIstore);
    free(ETAstore);
    free(ZETstore);

    free(XIItavg);
    free(ETAtavg);
    free(ZETtavg);

    free(Diss);
    free(Input);
    free(Energy);

    (cudaFree(d_Urms));
    (cudaFree(d_Vrms));
    (cudaFree(d_Wrms));
  
    fclose(stats);
    fclose(avgs);
    fclose(means);
    fclose(pdfs);
  }
  printf("stats free done \n"); 
  //Destroy fft plans
  (cufftDestroy(PlanZ2D));
  (cufftDestroy(PlanD2Z));
