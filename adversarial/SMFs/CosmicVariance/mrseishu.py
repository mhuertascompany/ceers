import numpy as np

def SQR(x):
	return x*x
def CUBE(x):
	return x*x*x
def POW4(x):
	return x*x*x*x

########################################################################

class EH:
	"""
	Eisenstein & Hu (1997) transfer function code
	Two changes I made to the distributed E&H code were to strip out
	code I don't use, and move the global variables into a structure
	that is explicitly passed.  This may have a deleterious impact on
	performance, but makes the routine more obviously thread safe
	(assuming the user takes appropriate care of the EH structure).
	
	Later, I optimized for speed, changing all of the math.h function
	calls to their floating point equivalents, and making floating
	point constants explicit to cut out type conversions.  I also added
	a new 'global' variables mrs_1_alpha_c, which is the reciprocal of
	alpha_c (to save that being calculated); I also created some
	temporary variable (mrstmp*) for optimizations.
	
	First call mrstfset to set the relevant cosmological parameters in
	the EH structure.  Then call mrstf to return T(k).  Note that I
	follow E&H units convention below: always Mpc, never h^-1 Mpc.
	"""
	
	########################################################################
	
	def __init__(self, omegam, omegab, h):
		"""
		omegam : float
		omegab : float
		h      : float
		"""
		omega0hh = omegam * h * h
		f_baryon = omegab / omegam
		Tcmb = 2.728 # hardwired here
		self.TFset_parameters(omega0hh, f_baryon, Tcmb)
	
	########################################################################
	
	def TFset_parameters(self, omega0hh, f_baryon, Tcmb):
		""" Set all the scalars quantities for Eisenstein & Hu 1997 fitting formula
		Note: Units are always Mpc, never h^-1 Mpc.
		omega0hh : float 
			The density of CDM and baryons, in units of critical dens, 
			multiplied by the square of the Hubble constant, in units of 100 km/s/Mpc
		f_baryon : float
			The baryon fraction of the total matter density
		Tcmb     : float
			The temperature of the CMB in Kelvin.  Tcmb<=0 forces use
			of the COBE value of  2.728 K. */
		Output   : 
				Nothing, but set many global variables used in TFfit_onek(). 
				You can access them yourself, if you want.
		"""
		assert(f_baryon > 0.0)
		assert(omega0hh > 0.0)
		
		omhh = omega0hh
		obhh = omhh*f_baryon
		if Tcmb<=0.0: Tcmb=2.728 # COBE FIRAS
		theta_cmb = Tcmb/2.7

		z_equality = 2.50e4*omhh/POW4(theta_cmb) # Really 1+z
		k_equality = 0.0746*omhh/SQR(theta_cmb)

		z_drag_b1 = 0.313 * omhh**(-0.419) * (1+0.607*omhh**0.674)
		z_drag_b2 = 0.238 * omhh**0.223
		z_drag = 1291*omhh**0.251 / (1+0.659*omhh**0.828) * (1+z_drag_b1*obhh**(z_drag_b2))

		R_drag     = 31.5 * obhh / POW4(theta_cmb) * (1000/(1+z_drag))
		R_equality = 31.5 * obhh / POW4(theta_cmb) * (1000/z_equality)

		sound_horizon = 2./3./k_equality * np.sqrt(6./R_equality) * np.log((np.sqrt(1+R_drag)+np.sqrt(R_drag+R_equality))/(1+np.sqrt(R_equality)))

		k_silk = 1.6 * obhh**0.52 * omhh**0.73 * (1+(10.4*omhh)**(-0.95))

		alpha_c_a1 = (46.9*omhh)**0.670 * (1+(32.1*omhh)**(-0.532))
		alpha_c_a2 = (12.0*omhh)**0.424 * (1+(45.0*omhh)**(-0.582))
		alpha_c = alpha_c_a1**(-f_baryon) * alpha_c_a2**(-CUBE(f_baryon))
		mrs_1_alpha_c = 1./alpha_c # introduced for optimization

		beta_c_b1 = 0.944 / (1+(458*omhh)**(-0.708))
		beta_c_b2 = (0.395*omhh)**(-0.0266)
		beta_c = 1.0/( 1+beta_c_b1*((1-f_baryon)**(beta_c_b2)-1) )

		y = z_equality/(1+z_drag)
		alpha_b_G = y * (-6.*np.sqrt(1+y) + (2.+3.*y)*np.log((np.sqrt(1+y)+1)/(np.sqrt(1+y)-1)))
		alpha_b = 2.07 * k_equality * sound_horizon * (1+R_drag)**(-0.75) * alpha_b_G

		beta_node = 8.41 * omhh**0.435
		beta_b = 0.5 + f_baryon + (3.-2.*f_baryon)*np.sqrt((17.2*omhh)**2+1)

		k_peak = 2.5*3.14159 * (1+0.217*omhh) / sound_horizon
		sound_horizon_fit = 44.5 * np.log(9.83/omhh) / np.sqrt(1+10.0*obhh**0.75)

		alpha_gamma = 1-0.328*np.log(431.0*omhh)*f_baryon + 0.38*np.log(22.3*omhh)* SQR(f_baryon)

		# set values in structure
		self.omhh = omhh
		self.obhh = obhh
		self.theta_cmb = theta_cmb
		self.z_equality = z_equality
		self.k_equality = k_equality
		self.z_drag = z_drag
		self.R_drag = R_drag
		self.R_equality = R_equality
		self.sound_horizon = sound_horizon
		self.k_silk = k_silk
		self.alpha_c = alpha_c
		self.mrs_1_alpha_c = mrs_1_alpha_c
		self.beta_c = beta_c
		self.alpha_b = alpha_b
		self.beta_b = beta_b
		self.beta_node = beta_node
		self.k_peak = k_peak
		self.sound_horizon_fit = sound_horizon_fit
		self.alpha_gamma = alpha_gamma
		
		# additional
		self.f_baryon = obhh/omhh
	
	########################################################################
	
	def mrstf(self, k, tf_baryon=None, tf_cdm=None):
		""" Returns the value of the full transfer function fitting formula.
		This is the form given in Section 3 of Eisenstein & Hu (1997).
		Notes: Units are Mpc, not h^-1 Mpc.
		tf_baryon, tf_cdm input value not used; replaced on output if the input was not NULL
		
		k         : float, wavenumber at which to calculate transfer function, in Mpc^-1
		tf_baryon : float, baryonic contribution to the full fit
		tf_cdm    : float, CDM contribution to the full fit
		output    : float
		"""
		
		# get values from structure
		#omhh              = self.omhh
		#obhh              = self.obhh
		#theta_cmb         = self.theta_cmb
		#z_equality        = self.z_equality
		k_equality        = self.k_equality
		#z_drag            = self.z_drag
		#R_drag            = self.R_drag
		#R_equality        = self.R_equality
		sound_horizon     = self.sound_horizon
		k_silk            = self.k_silk
		#alpha_c           = self.alpha_c
		mrs_1_alpha_c     = self.mrs_1_alpha_c
		beta_c            = self.beta_c
		alpha_b           = self.alpha_b
		beta_b            = self.beta_b
		beta_node         = self.beta_node
		#k_peak            = self.k_peak
		#sound_horizon_fit = self.sound_horizon_fit
		#alpha_gamma       = self.alpha_gamma
		f_baryon = self.f_baryon
		
		# vectorized
		k = np.atleast_1d(k)
		
		"""
		if k == 0.0:
			if tf_baryon is not None: tf_baryon = 1.0
			if tf_cdm    is not None: tf_cdm    = 1.0
			return 1.0
		"""
		
		k = np.abs(k) # Just define negative k as positive
		q = k/13.41/k_equality
		xx = k*sound_horizon

		T_c_ln_beta   = np.log(2.718282+1.8*beta_c*q)
		T_c_ln_nobeta = np.log(2.718282+1.8       *q)
		
		mrstmp1 = 386./(1.+69.9*q**1.08) # tmp variable for speed
		# T_c_C_alpha = 14.2/alpha_c + mrstmp1
		T_c_C_alpha   = 14.2*mrs_1_alpha_c + mrstmp1 # for optimization
		T_c_C_noalpha = 14.2               + mrstmp1

		# T_c_f = 1./(1.+POW4(xx/5.4))
		mrstmp2 = xx/5.4; # opt
		T_c_f = 1./(1.+mrstmp2*mrstmp2*mrstmp2*mrstmp2); # opt
		T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*SQR(q)) + (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*SQR(q))
		
		# s_tilde = sound_horizon * (1.0+CUBE(beta_node/xx))**-1./3.
		mrstmp3 = beta_node/xx # opt
		s_tilde = sound_horizon * (1.0+mrstmp3*mrstmp3*mrstmp3)**(-1./3.) # opt
		xx_tilde = k*s_tilde

		T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*SQR(q));
		# T_b = np.sin(xx_tilde)/xx_tilde * (T_b_T0/(1+SQR(xx/5.2)) + alpha_b/(1.+CUBE(beta_b/xx))*np.exp(-(k/k_silk)**1.4))
		mrstmp4 = xx/5.2 # opt
		mrstmp5 = beta_b/xx # opt
		T_b = np.sin(xx_tilde)/xx_tilde * (T_b_T0/(1+mrstmp4*mrstmp4) + alpha_b/(1.+mrstmp5*mrstmp5*mrstmp5)*np.exp(-(k/k_silk)**1.4)) # opt

		T_full = f_baryon*T_b + (1-f_baryon)*T_c

		"""
		# Now to store these transfer functions
		if tf_baryon is not None: tf_baryon = T_b
		if tf_cdm    is not None: tf_cdm    = T_c
		"""
		
		# vectorized
		T_full[k==0] = 1.0
		
		return T_full


