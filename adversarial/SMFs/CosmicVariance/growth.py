import numpy as np
from .qromb import qromb

class Growd:
	def __init__(self, omegam, omegal, omegak):
		"""
		omegam : double
		omegal : double
		omegak : double
		"""
		self.omegam = omegam
		self.omegal = omegal
		self.omegak = omegak
		# vectorized
		self.inva3h3_vec = np.vectorize(self.inva3h3) 
		self.dlin0 = qromb(self.inva3h3_vec, 0, 1)
	
	def inva3h3(self, a):
		"""
		a : float
		"""
		if a > 0:
			#ah = np.sqrt(self.omegam/a + self.omegal*a*a + self.omegak)
			#return 1./(ah*ah*ah)
			# optimized
			return (self.omegam/a + self.omegal*a*a + self.omegak)**(-3./2.)
		else:
			return 0.
	
	def dlina(self, a):
		"""
		a : float
		"""
		h = np.sqrt(self.omegam/(a*a*a) + self.omegal + self.omegak/(a*a))
		integral = qromb(self.inva3h3_vec, 0, a)
		return h*integral

