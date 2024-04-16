import numpy as np
from scipy import integrate

def fGR(a, omegam, omegav):
	""" function to be integrated
	a      : double
	omegam : double
	omegav : double
	output : double
	"""
	eta = np.sqrt(omegam/a + omegav*a*a + 1.0-omegam-omegav)
	f = 2.5/(eta*eta*eta)
	return f

def dplus(a, omegam, omegav):
	""" 
	a      : double
	omegam : double
	omegav : double
	output : double
	"""
	result, error = integrate.quad(fGR, 0, a, args=(omegam, omegav), epsabs=0, epsrel=1e-9, limit=10000)
	aux = np.sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav) # temp result 
	if a > 0:
		return result*aux/a;
	else:
		return 0

def adp(dpls, omegam, omegav):
	""" This inverts the linear growth factor
	dpls=dplus(a,omegam,omegav) for a=adp.
	dpls   : double
	omegam : double
	omegav : double
	output : double
	"""
	tol = 1.e-9
	
	# starting checks
	assert(dpls >= 0.), "error in adp, trying to invert a negative growth factor = %e . Stopping!" % dpls
	if dpls == 0:
		return 0
		
	# adown check
	adown = 0
	check = dplus(adown, omegam, omegav)
	assert(check <= dpls), "error in adp, trying to invert a growth factor = %e . Minimum not bracketed. Stopping!" % dpls
	
	# aup check
	aup = 1
	check = dplus(aup, omegam, omegav)
	assert(check >= dpls), "error in adp, trying to invert a growth factor = %e . Maximum not bracketed. Stopping!" % dpls
	
	# now we have bracketed the solution. Starting iteration
	error = abs(aup-adown)
	while error>tol:
		aguess = 0.5*(adown+aup)
		check = dplus(aguess, omegam, omegav)
		if check < dpls:
			adown = aguess
		else:
			aup = aguess
		error = abs(aup-adown)
	
	return aguess

