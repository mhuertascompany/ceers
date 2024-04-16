import numpy as np
from scipy.integrate import romberg

EPS_def  = 1e-6
JMAX_def = 20

#EPS_def  = 1e-5
#JMAX_def = 15

def qromb(func, a, b, EPS=EPS_def, JMAX=JMAX_def):
	""" Romberg integrator, based on Numerical Recipes qromb
	Returns the integral of the function func from a to b.
	Integration is performed by Romberg's method of order 2K, where, e.g., K=2 is Simpson's rule.
	divmax = n in wikipedia
	
	func : function taking float x
	a    : float, lower integration bound
	b    : float, upper integration bound
	EPS  : float, fractional accuracy desired, as determined by the extrapolation error estimate
	JMAX : int, limits the total number of steps
	K    : int, number of points used in the extrapolation, not used
	"""
	return romberg(func, a, b, rtol=EPS, divmax=JMAX, vec_func=True)
