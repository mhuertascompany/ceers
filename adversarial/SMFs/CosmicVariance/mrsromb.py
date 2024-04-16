import numpy as np
import copy

"""
Closed Romberg integrator, based on Numerical Recipes qromb

First I excised qromb from the NR library along with required
support routines, so that it can be compiled stand-alone.

The only change I made to the actual code is the passing through of
the void pointer fdata.  This is provided in case the user wants to
evaluate a function that depends on more than one variable.  This
is not the best-performing way to implement this, but it is
reasonably flexible.  An example is shown in exmrsromb.c.

mrsromb is generally thread safe itself.  However, it is the user's
responsibility to ensure the thread safety of the integrand
function, specifically through its potential reference to fdata.
"""

########################################################################

EPS  = 1.0e-5 # default was 1.0e-6
JMAX = 15     # default was 20
K    = 4

# look for Romberg integrator in scipy
# compare scipy romberg with scipy or wikipedia example

def mrsromb(func, a, b):
	""" qromb from NumRec
	
	Returns the integral of the function func from a to b.
	Integration is performed by Romberg's method of order 2K, where, e.g., K=2 is Simpson's rule.
	Parameters: 
	EPS is the fractional accuracy desired, as determined by the extrapolation error estimate
	JMAX limits the total number of steps
	K is the number of points used in the extrapolation.
	
	func  : function taking float x
	a     : float
	b     : float
	"""
	
	# These store the successive trapezoidal approximations
	s = np.zeros(JMAX)
	h = np.zeros(JMAX+1)
	
	h[0] = 1.0
	for j in range(JMAX):
		s[j] = trapzd(func, a, b, j)
		if j >= K:
			ss, dss = polint(h[j-K:j], s[j-K:j], 0.0)
			if abs(dss) <= EPS*abs(ss):
				return ss
		
		# The factor is 0.25 even though the stepsize is decreased by only 0.5. 
		# This makes the extrapolation a polynomial in h2, not just a polynomial in h.
		h[j+1] = 0.25*h[j]
		
	assert(False), "Too many steps in routine mrsromb"

########################################################################

def trapzd(func, a, b, n):
	""" Integration with trapezoidal rule, trapzd from NumRec
	func  : function taking float x
	a     : float, lower integration bound
	b     : float, upper integration bound
	n     : int, number of intervals for integration
	"""
	grid = np.linspace(a, b, n+1)
	data = func(grid)
	inte = np.trapz(data, grid)
	return inte

########################################################################

def polint(xa, ya, x):
	""" Polynomial interpolation, polint from NumRec (plus required elements of nrutils.c/h)
	Copied form "Numerical Recipes in FORTRAN 77"
	Given arrays xa and ya, each of length n, and given a value x, this routine
	returns a value y, and an error estimate dy. If P(x) is the polynomial of
	degree n−1 such that P(xai)=yai, i=1,...,n, then the returned value y=P(x).
	
	xa : array/float
	ya : array/float
	x  : float
	"""
	assert(len(xa) == len(ya))
	n = len(xa)
	
	# find the index ns of the closest table entry
	dif = np.abs(x-xa)
	ns = np.argmin(dif)
	dif = dif[ns]
	
	# initialize the tableau of c’s and d’s
	c = copy.deepcopy(ya)
	d = copy.deepcopy(ya)
	
	# initial approximation to y
	y = ya[ns]
	dy = 0.
	ns = ns-1
	
	for m in range(n-1):
		
		# for each column of the tableau, loop over the current c’s and d’s and update them
		ho = xa[:n-m-1]-x
		hp = xa[m+1:]-x
		w = c[1:n-m]-d[:n-m-1]
		den = ho-hp
		den = w/den
		d = hp*den
		c = ho*den
		
		# decide which correction, c or d, we want to add to our accumulating value of y, 
		# i.e., which path to take through the tableau—forking up or down. 
		# We do this in such a way as to take the most “straight line” route through the tableau 
		# to its apex, updating ns accordingly to keep track of where we are. 
		# This keeps the partial approximations centered (insofar as possible) on the target x. 
		if 2*(ns+1) < (n-m-1):
			dy = c[ns+1]
		else:
			dy = d[ns]
			ns = ns-1
		y += dy
	
	# The last dy added is the error indication.
	return y, abs(dy)


def polint_test(xa, ya, x):
	""" Polynomial interpolation, from numpy
	not exactly the same, but very similar
	"""
	assert(len(xa) == len(ya))
	n = len(xa)
	p = np.polyfit(xa, ya, deg=n-1)
	y = np.polyval(p,x)
	return y


"""
########################################################################
# Old

def polint(xa, ya, x):
	"" polint from NumRec (plus required elements of nrutils.c/h)
	
	Polynomial interpolation. Copied form "Numerical Recipes in FORTRAN 77"
	Given arrays xa and ya, each of length n, and given a value x, this routine
	returns a value y, and an error estimate dy. If P(x) is the polynomial of
	degree n−1 such that P(xai)=yai, i=1,...,n, then the returned value y=P(x).
	
	xa : array/float
	ya : array/float
	x  : float
	""
	assert(len(xa) == len(ya))
	n = len(xa)
	
	# find the index ns of the closest table entry
	dif = np.abs(x-xa)
	ns = np.argmin(dif)
	dif = dif[ns]
	
	# initialize the tableau of c’s and d’s
	c = copy.deepcopy(ya)
	d = copy.deepcopy(ya)
	
	# initial approximation to y
	y = ya[ns]
	dy = 0.
	ns = ns-1
	
	# for each column of the tableau, we loop over the current c’s and d’s and update them
	for m in range(n-1):
		for i in range(n-m-1):
			print(n,m,i)
			
			ho  = xa[i]-x
			hp  = xa[i+m+1]-x
			w   = c[i+1]-d[i]
			den = ho-hp
			if den == 0.0:
				print("Error in routine polint")
				print("This error can occur only if two input xa’s are (to within roundoff) identical")
				assert(False)
			den  = w/den
			d[i] = hp*den
			c[i] = ho*den
		
		if 2*(ns+1) < (n-m-1):
			dy = c[ns+1]
		else:
			dy = d[ns]
			ns = ns-1
		y += dy
	return y, dy

"""
