import numpy as np
from scipy import integrate

r"""
Computation of the variance (and covariance) of a galaxy field
using the two point correlation function
I = (1/Volume) * \int d^3x_1 d^3x_2 \xi(|x_1-x_2|
I = int (dx dy dz)/(2pi)^3  1/(1-cos(x)cos(y)cos(z))
"""

########################################################################
########################################################################

def integrate_MC(function, ranges, calls, args=()):
	""" Monte-Carlo integration
	http://barnesanalytics.com/monte-carlo-integration-in-python
	
	function : function with n parameters
	ranges   : 1darray/2-tuple/float
	calls    : int
	args     : additional function arguments (fixed)
	"""
	assert(np.shape(ranges)[1] == 2)
	assert(calls >= 1)
	
	n_dim = len(ranges)
	volume = np.array(ranges)
	volume = np.prod(volume[:,1]-volume[:,0])
	
	sampled = np.zeros((calls, n_dim))
	for i in range(n_dim):
		x1, x2 = ranges[i]
		sampled[:,i] = np.random.uniform(x1, x2, calls)
	
	# unvectorized, with args
	#values = np.zeros(calls)
	#for j in range(calls):
	#	values[j] = function(*sampled[j,:], *args)
	
	# vectorized, without args
	values = function(*sampled.T)
	
	integral = np.mean(values)                      *volume
	error    = np.std(values, ddof=1)/np.sqrt(calls)*volume
	return integral, error

########################################################################
########################################################################

class CorrFunc:
	def __init__(self, Ndat=None, xmin=None, xmax=None, delta=None, data=None, filename=None):
		""" 
		Ndat  : int
		xmin  : double
		xmax  : double
		delta : double
		data  : double
		filename : string
		"""
		#self.r0, self.gamma = 2.7, 1.8
		self.r0, self.gamma = 1.0, 1.0
		self.Ndat  = Ndat
		self.xmin  = xmin
		self.xmax  = xmax
		self.delta = delta
		self.data  = data
		if filename is not None:
			self.read(filename)
	
	def write(self, filename):
		""" Write the content of self into filename
		filename : string
		"""
		def formatted(value): return str(value)+'\n'
		with open(filename,'w') as f:
			# writes down header of cf
			f.write(formatted(self.Ndat))
			f.write(formatted(self.xmin))
			f.write(formatted(self.xmax))
			f.write(formatted(self.delta))
			# and these are the bunch of data 
			for d in self.data:
				f.write(formatted(d))
	
	def read(self, filename):
		""" Read the content of self from filename
		filename : string
		"""
		with open(filename,'r') as f:
			values = [float(line.rstrip()) for line in f]
		self.Ndat  = int(values[0])
		self.xmin  = values[1]
		self.xmax  = values[2]
		self.delta = values[3]
		self.data  = np.array(values[4:])
		self.radii = np.linspace(self.xmin, self.xmax, self.Ndat)
	
	def clean(self, filename, output):
		""" Clean the data in filename
		Replace negative values by interpolation from positive values, or set to zero
		filename : string, input filename
		output   : string, output filename
		"""
		self.read(filename)
		
		d = self.data
		r = self.radii
		delta = self.delta
		
		# compute second derivative
		deriv = np.zeros(len(d))
		deriv[1:-1] = (d[2:]+d[:-2]-2*d[1:-1])/(delta*delta)
		
		# find problematic points
		idx  = (r> 10) & (np.abs(deriv) > 0.5)
		idx |= (r> 25) & (np.abs(deriv) > 0.1)
		idx |= (r> 50) & (np.abs(deriv) > 0.02)
		idx |= (r>105) & (np.abs(deriv) > 0.004)
		
		# linearly extrapolate after each missing point
		for i in range(len(d)):
			if idx[i]:
				d[i] = d[i-2] + (d[i-1]-d[i-2]) * (r[i]-r[i-2]) / (r[i-1]-r[i-2])
		
		# remove large negative region at large r
		cut = np.where(d[~idx] < 0)[0][0]
		d[cut:] = 0
		
		self.data = d
		self.write(output)

########################################################################

R_CRIT = 3000.0

#@np.vectorize
def xi(r, cf):
	""" two point correlation function (analytical for the moment)
	r  : double
	cf : CorrFunc
	"""
	if r < R_CRIT:
	  return (cf.r0/r)**cf.gamma
	else:
		return 0
	return 0


#@np.vectorize # BUT fails sometimes and return integers ???
def xi_table(r, cf):
	""" two point correlation function (using stored data)
	r  : double
	cf : CorrFunc
	"""
	# min-max values check
	if r > cf.xmax:
		return 0
	elif r < cf.xmin:
		i = 0
	else:
		# look-up index for the correlation function table
		i = int(round((r-cf.xmin)/cf.delta))
	return cf.r0*cf.data[i] # /r*4.0 # temp fix

# vectorized
def xi_table_vec(r, cf):
	""" two point correlation function (using stored data)
	r  : array/double
	cf : CorrFunc
	"""
	r = np.atleast_1d(r)
	n = len(r)
	i   = np.zeros(n, dtype=int)
	out = np.zeros(n, dtype=float)
	idx_low = (r<=cf.xmax)
	idx_mid = idx_low & (r>=cf.xmin)
	i[idx_mid] = np.round((r[idx_mid]-cf.xmin)/cf.delta)
	out[idx_low] = cf.r0*cf.data[i[idx_low]] 
	return out


def g(k1, k2, k3, k4, k5, k6, cf):
	""" integration function (calls two point correlation function)
	k  : array/double
	cf : CorrFunc
	"""
	# computes distance between the two points (in 6dim notation)
	dist = np.sqrt((k1-k4)**2 + (k2-k5)**2 + (k3-k6)**2)
	return xi_table_vec(dist, cf)

########################################################################

def f_2p(x, r, cosmo):
	""" Two point correlation function integration 
	x : double
	r : double, radius
	cosmo : Cosmo
	"""
	q = x/(cosmo.h*cosmo.OmegaM)
	t = np.log(1.0+2.34*q) / (2.34*q) * (1.0+3.89*q+ (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
	f = x * (x*t*t) * np.sin(x*r)/r
	return f

def g_2p(x, r, cosmo):
	""" Sigma(M) function integration
	x : double
	r : double, radius
	cosmo : Cosmo
	"""
	xr = x*r
	q = x/(cosmo.h*cosmo.OmegaM)
	t = np.log(1.0+2.34*q) / (2.34*q) * (1.0+3.89*q+ (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
	w = 3.0*(np.sin(xr)/(xr*xr*xr) - np.cos(xr)/(xr*xr))
	g = x*x * (x*t*t) * w*w
	return g

########################################################################
########################################################################

def tpcf_init(filename, cosmo, clean=True):
	"""
	"""
	# min and max frequencies
	kmin = 0.0003
	#kmin = 0.00003
	kmax = 100.0
	
	# sigma_8 calculation
	rad = 8.0
	result, error = integrate.quad(g_2p, kmin, kmax, args=(rad,cosmo), epsabs=0, epsrel=1e-7, limit=1000)
	#ps_norm = cosmo.sigma8/result
	ps_norm = 1./result
	
	# start computing the 2point correlation function
	rad = 0.005
	
	cf = CorrFunc()
	cf.Ndat  = 30000
	cf.delta = 0.005
	cf.xmin  = 0.005
	cf.xmax  = cf.xmin+cf.delta*(cf.Ndat-1)
	cf.data = np.zeros(cf.Ndat, dtype=float)
	
	print("Radius Result")
	for i in range(cf.Ndat):
		rad = cf.xmin+i*cf.delta
		result, error = integrate.quad(f_2p, kmin, kmax, args=(rad,cosmo), epsabs=0, epsrel=1e-6, limit=100000)
		cf.data[i] = result*ps_norm
		print("%f %f" % (rad, result*ps_norm))
	print("Init TwoPoint Corr Func Done")
	
	# writes down the cf structure in a file
	cf.write(filename)
	
	# and now remormalize the data with correct sigma8
	ps_norm = cosmo.sigma8**2 * ps_norm
	cf.data *= ps_norm
	
	return cf

########################################################################

def cf_2p_read(filename, cosmo, beamdata):
	""" Reads two point correlation function
	"""
	
	# reads the cf structure in a file
	cf = CorrFunc(filename=filename)
	
	# now rescales the two point correlation function to the correct redshift
	norm = cosmo.growth(beamdata.Mean_z)*cosmo.sigma8
	cf.data *= norm*norm
	#print(" norm check ", norm)
	
	return cf

########################################################################

from functools import partial

def DM_variance(filename, cosmo, beamdata):
	""" DM variance within the pencil beam
	beamdata : Beamdata
	"""
	
	# initialize numerical two point correlation function (long computation)
	#cf = tpcf_init(filename, cosmo)
	
	# reads precomputed two point correlation function from file (fast)
	cf = cf_2p_read(filename, cosmo, beamdata)
	
	# assign limits
	xl = np.array([0]*6)
	xu = np.array([beamdata.DX, beamdata.DY, beamdata.DZ]*2)
	
	# normalization
	norm = abs(np.prod(xl-xu))

	# integration, unvectorized
	func   = g
	ranges = list(zip(xl,xu))
	calls  = 200000 # 200000
	args   = (cf,)
	
	# integration, vectorized
	# xi, xi_table must be vectorized
	func = partial(g, cf=cf)
	args = ()
	
	res, err = integrate_MC(func, ranges=ranges, calls=calls, args=args)
	err_norm = err/res
	res_norm = res/norm
	#print("MC integration: Mean = %f, Rel Error = %f " % (res_norm, err_norm))
	
	return res_norm

