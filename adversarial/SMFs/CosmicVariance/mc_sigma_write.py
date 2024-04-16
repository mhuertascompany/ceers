import numpy as np
import os

from .header import Cosmo
from .sigmaDM import tpcf_init, g, filename
from .tools import integrate_MC

def main():
	""" test main driver
	"""
	
	# COSMOLOGY
	#sigma_8 = 0.9
	#sigma_8 = 0.1941 # z=5.15
	#sigma_8 = 0.1675 # z=6.125
	sigma_8 = 0.1366 # z=7.75
	#sigma_8 = 0.1167 # z=9.25
	cosmo = Cosmo(OmegaM=0.28, h=0.7, sigma8=sigma_8)
	
	# initialize numerical two point correlation function
	overwrite = False
	if overwrite | (not os.path.isfile(filename)):
		tpcf_init(filename, cosmo)
	
	# ******** sigma11 *********
	#xl = [ 0, 0, 0, 0.0 ,0, 0 ]
	#xu = [ 6.0, 6.0, 320.0, 6.0, 6.0, 320.0 ]
	#xu = [ 6.0, 6.0, 320.0, 6.0, 6.0, 320.0 ]
	#xu = [ 24.0, 24.0, 20.0, 24.0, 24.0, 20.0 ]
	
	xl = np.array([ 0, 0, 0, 0 , 0, 0 ])
	#xu = [ 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 ]
	xu = np.array([ 19.5, 30.1, 342, 19.5, 30.1, 342 ])
	#xu = [ 1.7, 1.7, 269, 1.7, 1.7, 269 ]
	#xu = [ 6.0, 6.0, 320.0, 6.0, 6.0, 320.0 ]
	
	# normalization
	norm = abs(np.prod(xl-xu))
	
	# integrates
	calls = 200000 # 20000000
	res, err = integrate_MC(g, ranges=list(zip(xl,xu)), calls=calls, args=())
	err11_norm   = err/res
	sigma11_norm = res/norm
	print("Plain MC: Res = %f, Err = %f" % (sigma11_norm, err11_norm))
	
	# now start computing the correlation of nearby fields, we are
	# basically acting with bias set by the first parameter passed
	niter = 1
	skip  = 1.0
	
	# nearby fields correlation
	for jj in range(0,niter):
		
		xl[3] += skip
		xu[3] += skip
		
		# normalization
		norm = abs(np.prod(xl-xu))
		   
		# integrates
		res, err = integrate_MC(g, ranges=list(zip(xl,xu)), calls=calls, args=args)
		err12_norm   = err/res
		sigma12_norm = res/norm
		print("%f %f %f %f %f %f" % ((skip*(jj+1)-xu[3]+xl[3]), sigma12_norm/sigma11_norm, sigma12_norm, err12_norm, sigma11_norm, err11_norm))

