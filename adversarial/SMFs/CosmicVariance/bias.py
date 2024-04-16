import numpy as np

from .sigma2 import PK
from .growth import Growd

DELTA_CRIT    = 1.686
SQRT0707      = 0.840832920383116300 # np.sqrt(0.707)
MRS_1_SQRT2PI = 0.398942280401432678

########################################################################
########################################################################

class Bias:
	def __init__(self, m, z, growdat, ppk):
		""" Define common inputs for massfunc and bias
		z       : float
		growdat : Growd
		ppk     : PK
		"""
		if z > 0:
			dlin = growdat.dlina(1./(1.+z)) / growdat.dlin0
			w = DELTA_CRIT / dlin
			#print('dlin = ', dlin)
		elif z == 0:
			w = DELTA_CRIT
		else:
			assert(False), "stmassfunc error: z<0 is not allowed; z=%.2f" % z
		
		density = ppk.eh.omhh * 2.7755e11 # density at z=0 in Msun/Mpc^3
		scale = (3.*m/(4.*np.pi*density))**(1./3.)
		s = ppk.sigma2(scale) # long
		
		self.m       = m
		self.ppk     = ppk
		self.w       = w
		self.density = density
		self.scale   = scale
		self.s       = s
	
	def massfunc(self):
		""" Sheth-Tormen halo mass function, expressed as M*dn/dM(M,z)
		m : float
		"""
		m       = self.m
		ppk     = self.ppk
		w       = self.w
		density = self.density
		scale   = self.scale
		s       = self.s
		
		nut = SQRT0707 * w/np.sqrt(s)
		mdsdm = ppk.mdsigma2dm(scale)
		mdndm = density /m /s *(-mdsdm) *MRS_1_SQRT2PI *0.3222 *(1.+nut**-0.6) *nut *np.exp(-nut*nut/2.)
		return mdndm
	
	def bias(self, methodflag):
		""" Sheth-Tormen/Press-Schechter bias, expressed as M*dn/dM(M,z)
		methodflag : int
		"""
		w = self.w
		s = self.s
		
		nu = w*w/s
		# Press-Schechter
		if   methodflag == 0: 
			bias = 1. + (nu-1.)/DELTA_CRIT
		# Sheth-Tormen
		elif methodflag == 1: 
			bias = 1. + (SQRT0707*nu-1.)/DELTA_CRIT + 0.6/DELTA_CRIT/(1.+(SQRT0707*nu)**(0.3))
		return bias

########################################################################
########################################################################

def mean_bias(cosmo, indata, beamdata):
	""" Calculate the average bias of the sample
	
	Compute the minimum halo mass Mmin required to obtain the number 
	density of halos hosting the survey population. 
	Combining the halo filling factor and the intrinsic number of 
	objects in the survey (given the survey volume), we compute the 
	minimum halo mass in the Sheth & Tormen (1999) model required to 
	match the input number density.
	"""
	
	# mass function parameters
	z  = 0    # redshift to evaluate mass function
	nm = 400  # number of masses to sample mass function # 400 is necessary
	lgm1 = 8 
	lgm2 = 16 # mass function range in log_10 mass
	
	# parse input parameters
	omegam = cosmo.OmegaM
	omegal = cosmo.OmegaL
	omegab = cosmo.Omegab
	h      = cosmo.h
	sigma8 = cosmo.sigma8
	ns     = cosmo.ns
	z      = beamdata.Mean_z
	methodflag = indata.Bias
	
	# initialize transfer function and power spectrum (required to call sigma2)
	pk = PK(omegam, omegab, h, sigma8, ns)
	
	# set growth function data
	growdat = Growd(omegam, omegal, 1.-omegam-omegal)
	
	# average quantities
	delta_nh = 0
	cum_nh   = 0
	avg_bs   = 0
	
	# grid
	lgm = np.linspace(lgm2, lgm1, nm)
	m = 10**lgm
	
	for im in range(1,nm):
		#print(lgm[im])
		
		B = Bias(m[im], z, growdat, pk)
		mdndm = B.massfunc() # both as long
		bs    = B.bias(methodflag)
		
		# average quantities
		delta_nh = (lgm[im-1]-lgm[im])*mdndm
		cum_nh  += delta_nh
		avg_bs  += delta_nh*bs
		
		# check if correct bias has been found
		if mdndm >= beamdata.density:
			beamdata.set_addition(minmass=m[im], minbias=bs)
			return avg_bs/cum_nh
		
	print("Error in stmf.c bias not identified")
	print("This message probably appears because your number density of objects is too high")
	print("Only DarkMatterHalos with mass > 10^{%e} Msun/h are presently considered" % lgm1)
	print("This is to optimize the web browsing experience")
	print("Please file a bug ticket if you need to go further down in mass")
	return -1



"""
########################################################################
TRASH
########################################################################

# slower by factor 1.5, because repeat ppk.sigma2(scale)

def compute_w(z, growdat):
	""
	""
	if z > 0:
		dlin = growdat.dlina(1./(1.+z)) / growdat.dlin0()
		w = DELTA_CRIT / dlin
		#print('dlin = ', dlin)
	elif z == 0:
		w = DELTA_CRIT
	else:
		assert(False), "stmassfunc error: z<0 is not allowed; z=%.2f" % z
	return w

def massfunc(m, z, growdat, ppk):
	"" Sheth-Tormen halo mass function, expressed as M*dn/dM(M,z)
	m       : float
	z       : float
	growdat : Growd
	ppk     : PK
	""
	w = compute_w(z, growdat)
	density = ppk.eh.omhh * 2.7755e11 # density at z=0 in Msun/Mpc^3
	scale = (3.*m/(4.*np.pi*density))**(1./3.)
	s = ppk.sigma2(scale)
	nu = w/np.sqrt(s)
	nut = SQRT0707 * nu

	mdsdm = ppk.mdsigma2dm(scale)
	mdndm = density /m /s *(-mdsdm) *MRS_1_SQRT2PI *0.3222 *(1.+nut**-0.6) *nut *np.exp(-nut*nut/2.)
	return mdndm

def bias(m, zz, growdat, ppk, methodflag):
	"" Sheth-Tormen/Press-Schechter bias, expressed as M*dn/dM(M,z)
	m          : float
	zz         : float
	growdat    : Growd
	ppk        : PK
	methodflag : int
	""
	
	# correction
	z = zz+0.
	corr = growdat.dlina(1./(1.+z)) / growdat.dlina(1./(1.+zz))
	#print(corr)
	
	w = compute_w(z, growdat)
	density = ppk.eh.omhh * 2.7755e11 # density at z=0 in Msun/Mpc^3
	scale = (3.*m/(4.*np.pi*density))**(1./3.)
	s = ppk.sigma2(scale)
	nu = w*w/s
	
	# Press-Schechter
	if   methodflag == 0: 
		bias = 1. + (nu-1.)/DELTA_CRIT
	# Sheth-Tormen
	elif methodflag == 1: 
		bias = 1.
		bias += corr * (SQRT0707*nu-1.)/DELTA_CRIT
		bias += corr * 0.6/DELTA_CRIT/(1.+(SQRT0707*nu)**(0.3))
	
	return bias

########################################################################

def bias():
	z = zz+0.
	corr = growdat.dlina(1./(1.+z)) / growdat.dlina(1./(1.+zz))
	#print(corr)
	
	bias = 1.
	bias += corr * (SQRT0707*nu-1.)/DELTA_CRIT
	bias += corr * 0.6/DELTA_CRIT/(1.+(SQRT0707*nu)**(0.3))

########################################################################

# old bits

def bias_stmf(cosmo, indata, beamdata):
	""
	""
	# mass function parameters
	z  = 0    # redshift to evaluate mass function
	nm = 400  # number of masses to sample mass function
	lgm1 = 8
	lgm2 = 16 # mass function range in log_10 mass
	
	# average quantities
	avg_bs   = 0
	cum_nh   = 0
	delta_nh = 0
	lgm_old  = lgm2
	
	for im in range(nm, -1, -1):
		print(im)
		
		lgm = im*(lgm2-lgm1)/(nm-1)+lgm1
		m = 10**lgm
		
		mdndm = stmdndm(m, z, growdat, pk) # both as long
		bs = stbias(m, z, growdat, pk, methodflag)
		
		# average quantities
		delta_nh = (lgm_old-lgm)*mdndm
		cum_nh  += delta_nh
		avg_bs  += delta_nh*bs
		lgm_old  = lgm
"""
