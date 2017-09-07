____________________________________________________________________________________________________________________________


		GPU Pseudospectral solver for the triply periodic 3D incompressible Navier Stokes equations 

		For referencing please see the following paper:

		"Sustaining processes via recurrent flows in body-forced turbulence"
		Lucas & Kerswell, J. Fluid Mech. (2017)
____________________________________________________________________________________________________________________________


Outline of the code:

	   Code is of a standard pseudospectral form where timestepping occurs on fourier coefficients of vorticity
	   with nonlinear convolutions computed in physical space. Timestepping scheme is Heun's method
	   with Crank-Nicolson on diffusion terms. 
	   
	   Code is for use on NVIDIA GPUs via the CUDA programming API.
	   
	   Use is made of the CUFFT library for the Fast Fourier Transform (FFT) on GPUs. 

	   All DNS routines are found in src/ and include various diagnostic functions and
	   recurrence searching routines for finding close cycles of dynamics for convergence to unstable
	   periodic orbits. 

	   Also included is the Newton-GMRES-hookstep code for converging nonlinear unstable solutions, 
	   inluding periodic orbits. See src_gmres for the source code. 

	   Makefiles are included but certain environment variables are required to link to the libs
____________________________________________________________________________________________________________________________

bin/ contains input data, scripts and a short readme for setting up a job.
Please feel free to contact me for assistance or advice on using the code.

  -----------
  Contact
  -----------

        Authors: Dan Lucas
        Institution: DAMTP, University of Cambridge
        Email:  dl549@cam.ac.uk
	
August 2017
