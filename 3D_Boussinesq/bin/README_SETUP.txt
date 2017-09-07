
		GPU Pseudospectral solver for the triply periodic 3D boussinesq equations with background linear stratification

		For referencing please see the following papers:

		"Layer formation in horizontally forced stratified turbulence: connecting exact coherent structures to linear instabilities"
		Lucas, Caulfield & Kerswell, J. Fluid Mech. (2017) https://arxiv.org/abs/1701.05406

		"Mixing by unstable periodic orbits in buoyancy dominated stratified turbulence"
		Lucas & Caulfield, J. Fluid Mech. (2017) https://arxiv.org/abs/1706.02536

____________________________________________________________________________________________________________________________

Setting up a job:

	   Resolution and geometry is set at compile time in src/global_params.F90

	   Input files are params.txt for DNS and additional paramsEXTRA.txt for GMRES
	   plus an optional initial condition file, e.g. Zk.in

	   Expected directory structure for DNS is JOB0/params.txt and JOB1/params.txt
	   for the MPI parallel run, or just JOB0 for a serial run, so that JOB* is the 
	   working directory. 

	   For GMRES working directory is the current one and an input file is NOT optional
____________________________________________________________________________________________________________________________

Runtime parameters:

	params.txt

	    ISEED:	random number seed for IC, ensure well separated for each case
	    IREST: 	restart index from file for files with multiple state vectors (if 0 a random IC is generated)
	    infile:	file name/path for input IC
	    infile type: type of input file (A-H) see init.F90 for differences (e.g. UPO or DNS output)
	    KF:	   	forcing wavenumber, i.e. f=sin(KF*y)
	    Re:		Reynolds number
	    Ri:		Buoyancy parameter (Richardson number IF theta=0, denoted "B" in papers above)
	    Theta:	Inclination angle of the gravity vector in degrees: 0 means vertical shear, 90 means horizontal shear
	    Sc:		Schmidt or Prandtl number
	    TSTEP:	Total simulation duration in non-dimensional units
	    ResidualThreshold: Threshold for the recurrence residual when seeking guesses (RCFLAG=1)
	    outfreq1:	frequency of high frequency outputs
	    outfreq2:	frequency of low frequency outputs
	    DELT:	timestep
	    AMPV:	rescaling of amplitude of random initial condition 
	    STATSFLAG:  flag to determine wether diagnostics are computed
	    RCFLAG:	flag to determine wether recurrences are sought
	    UPOflag:	flag to determine wether input is a UPO or not (reads TSTEP etc. from infile)
	    ADAPTflag:	flag to determine wether to use adaptive timesteps (set to 0 for GMRES runs)
	    ISTOP:	index to stop processing input states from infile (for sweeping through continuation run)
	    Dtarget:	Target dissipation rate for throttling method (set to 0 for unthrottled)

____________________________________________________________________________________________________________________________

         paramsEXTRA.txt

	    IGUESSEND:  when running through a file of guesses, sets how many from IREST to process, or the  number of continuation steps
	    maxPERIOD:  filter any guesses with periods above this
	    gmres ResidualThreshold:   filter any guesses with residual above this
	    arclcFLAG:  flag to determine wether we're doing continuation 
	    1stOrder:	flag to choose first order extrapolation of contination
	    2ndOrder PredOnly:	flag to choose second order prediction for extrapolation of contination
	    Full 2ndOrder:	flag to choose full second order extrapolation of contination
	    drN_reset:	flag to reset curvature prediction of solution branch (usually leave F)
	    Fixed Period: flag to force fixed period for steady states (period is not converged)
	    Fixed Period Size: length of trajectory when period is fixed (usually small)
	    delta_v2:	 starting step of parameter along solution continuation
	    ArnoldiFLAG: flag to determine wether to do Arnoldi stability calculation (needs some compile time changes in source for resolution)
	    ParFLAG:	 Chooses which parameter we're continuing in (1: 1/Re, 2: Ri, 3: alpha)
	    
  -----------
  Contacts
  -----------

        Authors: Dan Lucas
        Institution: DAMTP, University of Cambridge (Now Keele)
        Email:  dl549@cam.ac.uk (Now d.lucas1@keele.ac.uk)
	
August 2017
