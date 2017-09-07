____________________________________________________________________________________________________________________________


		Newton-GMRES-Hookstep code for computing unstable relative periodic orbits

		For referencing please see the following papers:

		"Simple invariant solutions embedded in 2D Kolmogorov turbulence"
		Chandler & Kerswell, J. Fluid Mech. (2013)

		"Sustaining processes via recurrent flows in body-forced turbulence"
		Lucas & Kerswell, J. Fluid Mech. (2017)
____________________________________________________________________________________________________________________________


Outline of the code:

	   Code is a little less well maintained than the Boussinesq version. I haven't had time to comment carefully or optimise the code for readability so please just email with questions and look at the other versions here. One important point is that the resolution requires setting in global_params.F90 AND in kernels.h for the GMRES side and the CUDA-timestepper side. 

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
