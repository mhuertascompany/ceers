import numpy as np

from .mrseishu import EH
from .qromb import qromb
from .bessel import NBESSEL, besselzero

SIG2EPS    = 1e-5
UPPERZERO  = 30
MRS_1_2PI2 = 0.05066059182116889 # 1/(2 pi^2)

########################################################################

def bessel_function_full(x):
	""" Spherical bessel function (full)
	x : array/float
	"""
	w = 3.*(x*np.cos(x) - np.sin(x))/(x*x*x)
	return w*w

def bessel_function_expansion(x):
	""" Spherical bessel function (polynomial expansion)
	this replacement actually speeds this subroutine up a lot, also in Python
	x : array/float
	"""
	x2  = x*x
	x4  = x2*x2
	x6  = x4*x2
	x8  = x4*x4
	x10 = x6*x4
	return 1. - x2*0.2 + 3.*x4/175. - 4.*x6/4725. + 2.*x8/72765. - x10/1576575.

def trig_function_full(x):
	""" Trig function (full)
	optimized, especially for arrays
	x : array/float
	"""
	xcosx = x*np.cos(x)
	sinx = np.sin(x)
	x2 = x*x
	return -6. * (xcosx-sinx) * (3.*xcosx + (x2-3.)*sinx) / (x2*x2*x2)

def trig_function_expansion(x):
	""" Trig function (polynomial expansion)
	x : array/float
	"""
	x2  = x*x
	x4  = x2*x2
	x6  = x4*x2
	x8  = x4*x4
	x10 = x6*x4
	return -2.*x2/15 + 4.*x4/175 - 8.*x6/4725 + 16.*x8/218295 - 2.*x10/945945

########################################################################

class PK:
	def __init__(self, omegam, omegab, h, sigma8, ns):
		""" initialize both transfer function and P(k)
		omegam : float
		omegab : float
		h      : float
		sigma8 : float
		"""
		self.setnorm = False
		self.eh = EH(omegam, omegab, h)
		self.normsig8(sigma8, h, ns)
		
	def normsig8(self, sigma8, h, ns):
		"""
		h      : float
		sigma8 : float
		ns     : float
		"""
		assert(not self.setnorm), "P(k) parameters already set (or bad PK initialization)"
		
		self.setnorm = True
		scale8 = 8./h # 8 h^-1 Mpc
		self.ns = ns
		self.anorm = 1e7 # dummy value, but keeps the sigma_8 integral around unity
		sig2 = self.sigma2(scale8)
		self.anorm *= sigma8*sigma8/sig2
		self.sig8 = self.sigma2(scale8)

	def powerk(self, k): # not used
		""" Power spectrum P(k) 
		k   : float
		"""
		assert(self.setnorm), "Must set P(k) parameters before calling sigma2"
		
		anorm = self.anorm
		ns = self.ns
		tf = self.eh.mrstf(k)
		
		# P(k) A     * k^ns  * T(k)^2
		return anorm * k**ns * tf*tf

	########################################################################

	def sigma2(self, scale):
		"""
		scale : float
		"""
		assert(self.setnorm), "Must set P(k) parameters before calling sigma2"
		
		self.scale = scale
		
		lnkeq   = np.log(self.eh.k_equality) - 3
		lnkwlow = np.log(1/scale) - 4
		lnk1 = np.min([lnkeq,lnkwlow])

		upperzero = UPPERZERO
		lnk2 = np.log(besselzero[upperzero]/scale)

		#print("lnkeq: %g lnkwlow: %g lnk1: %g lnk2: %g" % (lnkeq,lnkwlow,lnk1,lnk2))
		#ncall = 0
		#ncall2 = 0

		sigma2 = qromb(self.dsigtopdlnk, lnk1, lnk2)
		totsig = sigma2
		
		#print("init integral from %g to %g  sigma2: %g" % (lnk1,lnk2,sigma2))
		#print("number of integrand calls: %d" % ncall2)

		flag_trylow  = True
		flag_tryhigh = True

		lnklow2  = lnk1 # set upper limit of lower expansion to initial lower limit
		lnkhigh1 = lnk2 # set lower limits of upper expansion to initial upper lim

		# expansion integral loop
		while flag_trylow or flag_tryhigh:
			#print(flag_trylow, flag_tryhigh)
			if flag_trylow:
				lnklow1 = lnklow2 - 1 # use \Delta lnk = 1
				#ncall2 = 0
				sigma2 = qromb(self.dsigtopdlnk, lnklow1, lnklow2)
				totsig += sigma2
				#print("low exp from %g to %g  sigma2: %g  totsig: %g" % (lnklow1,lnklow2,sigma2,totsig))
				#print("number of integrand calls: %d\n",ncall2)
				
				lnklow2 = lnklow1 # reset upper limit for next time
				if sigma2 < SIG2EPS*totsig:
					flag_trylow = False

			if flag_tryhigh:
				upperzero += 1
				assert(upperzero < NBESSEL), "Need more bessel zeros"
				
				lnkhigh2 = np.log(besselzero[upperzero]/scale) # int to next zero
				#ncall2 = 0
				sigma2 = qromb(self.dsigtopdlnk, lnkhigh1, lnkhigh2)
				totsig += sigma2
				#print("high exp from %g to %g sigma2: %g totsig: %g" % (lnkhigh1,lnkhigh2,sigma2,totsig))
				#print("number of integrand calls: %d" % ncall2)
				lnkhigh1 = lnkhigh2 # reset lower limit for next time
				if sigma2 < SIG2EPS*totsig:
					flag_tryhigh = False
		
		#print("total lnk range: %g to %g  max upperzero: %d" % (lnklow1,lnkhigh2,upperzero))
		#print("total number of integrand calls: %d" % ncall)
		return totsig

	def dsigtopdlnk(self, lnk):
		"""
		Integrand of spherical top-hat density variance integral, expressed
		per ln k interval, as a function of ln k. 

		The polynomial approximation to the window function at small k is
		required for accuracy (presumably it is also much faster).  I have
		verified that it returns identical results to the proper Bessel
		function value over the range I use it, at the precision of float
		(i.e., if you make it double then you need to recheck this).
		
		lnk : float
		"""
		
		# counter of number of calls
		#ncall += 1
		#ncall2 += 1
		
		# vectorized
		lnk = np.atleast_1d(lnk)
		
		k = np.exp(lnk)
		x = self.scale * k
		tf = self.eh.mrstf(k)
		
		idx = (x<1)
		w2 = np.zeros_like(k)
		w2[ idx] = bessel_function_expansion(x[idx])
		w2[~idx] = bessel_function_full(x[~idx])
		
		anorm = self.anorm
		ns    = self.ns
		
		#      1/(2 pi^2) * P(k)                   * W(k)^2 * k^2 * dk/dlnk
		#      1/(2 pi^2) * A     * k^ns  * T(k)^2 * W(k)^2 * k^2 * k
		return MRS_1_2PI2 * anorm * k**ns * tf*tf  * w2     * k*k * k

	########################################################################
	
	def mdsigma2dm(self, scale):
		"""
		scale : float
		"""
		assert(self.setnorm), "Must set P(k) parameters before calling dsigma2dm"
		
		self.scale = scale
		
		#lnkeq   = np.log(self.eh.k_equality) - 3
		lnkwlow = np.log(1/scale) + 1
		lnk1 = lnkwlow
		
		upperzero = UPPERZERO
		lnk2 = np.log(besselzero[upperzero]/scale)

		#print("lnkeq: %g lnkwlow: %g lnk1: %g lnk2: %g" % (lnkeq,lnkwlow,lnk1,lnk2))
		#ncall  = 0
		#ncall2 = 0

		dsigma2dm = qromb(self.ddsigtopdmdlnk, lnk1, lnk2)
		totsig = dsigma2dm
		
		#print("init integral from %g to %g  dsigma2dm: %g" % (lnk1,lnk2,dsigma2dm))
		#print("number of integrand calls: %d" % ncall2)

		flag_trylow  = True
		flag_tryhigh = True

		lnklow2  = lnk1 # set upper limit of lower expansion to initial lower limit
		lnkhigh1 = lnk2 # set lower limits of upper expansion to initial upper lim

		# expansion integral loop
		while flag_trylow or flag_tryhigh:
			#print(flag_trylow, flag_tryhigh)
			if flag_trylow:
				lnklow1 = lnklow2 - 4 # use \Delta lnk = 1, optimized with 4
				#ncall2 = 0
				dsigma2dm = qromb(self.ddsigtopdmdlnk, lnklow1, lnklow2)
				totsig += dsigma2dm
				#print("low exp from %g to %g  dsigma2dm: %g  totsig: %g" % (lnklow1,lnklow2,dsigma2dm,totsig))
				#print("number of integrand calls: %d" % ncall2)
				lnklow2 = lnklow1 # reset upper limit for next time
				if -dsigma2dm < -SIG2EPS*totsig:
					flag_trylow = False

			if flag_tryhigh:
				upperzero += 1
				assert(upperzero < NBESSEL), "Need more bessel zeros"

				lnkhigh2 = np.log(besselzero[upperzero]/scale) # int to next zero
				# ncall2 = 0
				dsigma2dm = qromb(self.ddsigtopdmdlnk, lnkhigh1, lnkhigh2)
				totsig += dsigma2dm
				#print("high exp from %g to %g dsigma2dm: %g totsig: %g" % (lnkhigh1,lnkhigh2,dsigma2dm,totsig))
				#printf("number of integrand calls: %d" % ncall2)
				lnkhigh1 = lnkhigh2 # reset upper limit for next time
				if -dsigma2dm < -SIG2EPS*totsig:
					flag_tryhigh = False

		#print("total lnk range: %g to %g max upperzero: %d" % (lnklow1,lnkhigh2,upperzero))
		#print("total number of integrand calls: %d" % ncall)

		return totsig
	
	
	def ddsigtopdmdlnk(self, lnk):
		"""
		Integrand of derivative of spherical top-hat density (w.r.t. mass,
		but see below) variance integral, expressed per ln k interval, as a
		function of ln k.

		The derivative introduces a factor of mass into the integrand.
		However, since this doesn't depend on k it can be moved outside of
		the integral, which I have done.  All the other factors are kept in
		here.

		The polynomial approximation to the window function at small k is
		required for accuracy, and is good to better than 1e-9 accuracy.
		
		lnk : float
		"""
		# counter of number of calls
		#ncall  += 1
		#ncall2 += 1
		
		k = np.exp(np.atleast_1d(lnk))
		x = self.scale * k
		tf = self.eh.mrstf(k)
		
		# d/dM = dx/dR dR/dM d/dx = k R/3M d/dx = x/3M d/dx
		# wfact = x/3 * d/dx W^2(x) = 2x/3 W(x) dW/dx
		
		wfact = np.zeros_like(k)
		idx = (x<0.25)
		wfact[ idx] = trig_function_expansion(x[idx])
		wfact[~idx] = trig_function_full(x[~idx])
		
		anorm = self.anorm
		ns = self.ns
		#      1/(2 pi^2) * P(k)              * x/3*2*W(x)*dW/dx * k^2 * dk/dlnk
		return MRS_1_2PI2 * anorm*k**ns*tf*tf * wfact            * k*k * k

