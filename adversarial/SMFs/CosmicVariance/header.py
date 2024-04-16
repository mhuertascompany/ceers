import numpy as np
from scipy import integrate

from .growth_factor import dplus

class Cosmo:
	def __init__(self, OmegaL=0.7, OmegaM=0.3, Omegab=0.0469, h=0.7, sigma8=0.9, ns=1):
		""" 
		OmegaL : float, dark energy     abundance parameter
		OmegaM : float, total matter    abundance parameter
		Omegab : float, baryonic matter abundance parameter
		h      : float, Hubble expansion rate parameter, with H0 = h*100 km/s/Mpc
		sigma8 : float, root-mean-square matter fluctuation averaged over a sphere of radius 8 Mpc/h
		ns     : float, scalar spectral index 
		"""
		assert(OmegaM >= 0)
		assert(Omegab >= 0)
		assert(h >= 0)
		assert(sigma8 > 0)
		
		self.OmegaL = OmegaL
		self.OmegaM = OmegaM
		self.Omegab = Omegab
		self.Omegak = 1.-OmegaL-OmegaM
		self.h      = h
		self.H0     = 100.*h # [km/s/Mpc]
		self.sigma8 = sigma8
		self.ns     = ns
		self.c      = 2.99792458e5 # speed of light [km/s]
		
		# pre-computation for the scaling of self.growth, faster
		self.ref = dplus(1., self.OmegaM, self.OmegaL)
		self.Dgrowth0 = 1
		self.Dgrowth0 = self.growth_test(0)

	def e(self, z):
		""" E(z) with H(z) = H0*E(z)
		z : double
		"""
		return np.sqrt(self.OmegaM*(1+z)**3 + self.OmegaL + self.Omegak*(1+z)**2)
	
	def inv_e(self, z):
		""" 1/E(z) with H(z) = H0*E(z)
		z : double
		"""
		return 1./self.e(z)
	
	def H(self, z): 
		""" H(z) = H0*E(z) [km/s/Mpc]
		z : double
		"""
		return H0*self.e(z)
	
	def comoving_dist(self, z):
		""" Comoving distance (distance between z=0 and input redshift) [Mpc/h]
		z : double
		"""
		assert(z >= 0.), "Error, negative input redshift in comoving_dist"
		grid = np.linspace(0, z, 4000)
		data = self.inv_e(grid)
		integ = np.trapz(data, grid)*self.c/self.H0 *self.h
		return integ
	
	def growth(self, z):
		""" Linear growth factor, dimensionless
		z      : double
		output : double
		"""
		a = 1./(1.+z)
		result = dplus(a, self.OmegaM, self.OmegaL)
		result = result/self.ref
		return result
	
	def growth_test(self, z):
		""" Linear growth factor, dimensionless
		first "a" is to scale the growth factor D = g*a
		about 1.5 times faster than the other growth function
		z      : double
		output : double
		"""
		a = 1./(1.+z)
		eps = a/1e4
		agrid = np.linspace(eps, a, 1000) # 1000 is far enough
		zgrid = 1./agrid-1.
		return 1./self.Dgrowth0 * a * 5.*self.OmegaM/2. * self.e(z)/a * integrate.simps(1./(agrid*self.e(zgrid))**3, agrid)

########################################################################

class Indata:
	def __init__(self, SurveyAreaX, SurveyAreaY, MeanZ, Dz, IntNum=1, 
		         HaloOff=1.0, Completeness=1.0, Bias=1):
		"""
		SurveyAreaX  : float, FoV_XSize (arcmin)
		SurveyAreaY  : float, FoV_YSize (arcmin)
		MeanZ        : float, Average Redshift
		Dz           : float, Redshift Width
		IntNum       : int, Intrinsic Number of Objects
		HaloOff      : float, Average Halo Occupation
		Completeness : float, Completeness
		Bias         : int, Method, 0:Press-Schechter, 1:Sheth-Tormen
		"""
		self.SurveyAreaX  = SurveyAreaX
		self.SurveyAreaY  = SurveyAreaY
		self.MeanZ        = MeanZ
		self.Dz           = Dz
		self.IntNum       = IntNum
		self.HaloOff      = HaloOff
		self.Completeness = Completeness
		self.Bias         = Bias
		self.validation()
	
	def validation(self):
		"""
		"""
		valid_flag = 1
		allsky = 21600.0
		
		if (self.SurveyAreaX<=0.0) or (self.SurveyAreaX>allsky):
			print("SurveyAreaX [arcmin] = %e is invalid" % self.SurveyAreaX)
			valid_flag = 0
		
		if (self.SurveyAreaY<=0.0) or (self.SurveyAreaY>allsky):
			print("SurveyAreaY [arcmin] = %e is invalid" % self.SurveyAreaY)
			valid_flag = 0
		
		if (self.MeanZ<=0):
			print("Mean redshift = %e is invalid" % self.MeanZ)
			valid_flag = 0
		
		if (self.Dz<=0) or (self.Dz>(2.0*self.MeanZ)):
			print("Delta redshift = %e is invalid" % self.Dz)
			valid_flag = 0
		
		if (self.IntNum<=0):
			print("Number of objects = %e is invalid" % self.IntNum)
			valid_flag = 0
		
		if (self.HaloOff<=0) or (self.HaloOff>1.0):
			print("Halo Occupation Factor = %e is invalid" % self.HaloOff)
			valid_flag = 0
		
		if (self.Completeness<=0) or (self.Completeness>1.0):
			print("Completeness Factor = %e is invalid" % self.Completeness)
			valid_flag = 0
		
		if (self.Bias not in [0,1]):
			print("Bias = %e is invalid" % self.Bias)
			valid_flag = 0
		
		if valid_flag==0:
			print("Input parameters did not pass validation check")
			print("Please double check them and try again")
			print("If you still think that this error message should not appear, file a bug ticket with Michele")
			assert(False)

########################################################################

class Beamdata:
	def __init__(self, indata, cosmo):
		""" Tranforms input data into beam data
		"""
		# average distance (to compute DX and DY) [Mpc/h]
		dist = cosmo.comoving_dist(indata.MeanZ)
		self.DX = indata.SurveyAreaX/60./360.*6.2831853*dist
		self.DY = indata.SurveyAreaY/60./360.*6.2831853*dist
		
		# beam length (DZ) [Mpc/h]
		dist  = cosmo.comoving_dist(indata.MeanZ-0.5*indata.Dz)
		dist1 = cosmo.comoving_dist(indata.MeanZ+0.5*indata.Dz)
		self.DZ = dist1-dist
		
		# beam mean redshift
		self.Mean_z = indata.MeanZ
		
		# number density of dark matter halos
		self.HaloOff = indata.HaloOff
		self.set_density(indata.IntNum)
		
		self.minmass = -99
		self.minbias = -99
	
	def set_density(self, N):
		""" Set the Intrinsic Number Density for Dark Matter Halos Hosts [(Mpc/h)^-3]
		N : float, number of galaxies
		"""
		self.density = N/self.HaloOff /(self.DX*self.DY*self.DZ)
	
	def set_addition(self, minmass, minbias):
		""" Save extra data, from original addition structure
		minmass : float, minimum DM halo mass to be in the sample
		minbias : float, bias @ minmass
		"""
		self.minmass = minmass
		self.minbias = minbias

