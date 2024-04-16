import numpy as np
import os

from .header import Cosmo, Indata, Beamdata
from .sigmaDM import DM_variance, tpcf_init
from .bias import mean_bias

########################################################################
"""
Engine for the CosmicVarianceCalculator by Michele Trenti & Massimo
Stiavelli written by Michele Trenti, September 2007

This program computes the total fractional uncertainty in the number
counts of a sample of high redshift LymanBreakGalaxies for a given
pointing.

The dark matter two point correlation function is fixed for the moment
to minimize cpu time usage.
"""
########################################################################

def get_filename(clean=True):
	""" Path to the sigmaDM file
	clean : bool, if True, use the cleaned version of the file
	"""
	# directory of this file, independently of the file importing the class
	#directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
	directory = os.path.abspath(os.path.dirname(__file__)) + os.sep # identical so far
	if clean:
		return directory + "cf2pt_cleaned.dat"
	else:
		return directory + "cf2pt.dat"

########################################################################

class CosmicVariance:
	def __init__(self, indata, cosmo={}):
		"""
		indata : dic
		cosmo  : dic 
		"""
		self.indata   = Indata(**indata)
		self.cosmo    = Cosmo(**cosmo)
		self.beamdata = Beamdata(self.indata, self.cosmo)
		self.set_sigmaDM  = False
		self.set_meanbias = False
	
	def initialize_sigmaDM(self):
		""" long
		"""
		tpcf_init(get_filename(clean=False), self.cosmo, clean=clean)
	
	def clean_sigmaDM(self):
		"""
		"""
		from .sigmaDM import CorrFunc
		filename = get_filename(clean=False)
		output   = get_filename(clean=True)
		cf = CorrFunc(filename=filename)
		cf.clean(filename=filename, output=output)
	
	def compute_sigmaDM(self, clean=True):
		""" Compute the dark matter cosmic variance
		N is not necessary
		clean : bool
		"""
		filename = get_filename(clean=clean)
		assert(os.path.isfile(filename)) # Initialize/clean the filename first
		self.sigma2DM = DM_variance(filename, self.cosmo, self.beamdata)
		self.set_sigmaDM = True
	
	def compute_meanbias(self, N=None):
		""" Compute the mean galaxy bias
		Can modify number and completeness
		N : float, number of galaxies
		"""
		if N is not None:
			self.indata.IntNum = N
			self.beamdata.set_density(N)
		self.meanbias = mean_bias(self.cosmo, self.indata, self.beamdata)
		self.set_meanbias = True
	
	def compute_cosmicvariance(self):
		""" Compute the sample cosmic variance (relative)
		estimate of the error to be implemented
		"""
		sigma2DM = self.sigma2DM
		meanbias = self.meanbias
		return meanbias*meanbias*sigma2DM
	
	def write(self):
		"""
		"""
		assert(self.set_sigmaDM and self.set_meanbias)
		
		cosmo  = self.cosmo
		indata = self.indata
		beamdata = self.beamdata
		meanbias = self.meanbias
		sigma2DM = self.sigma2DM
		
		# compute total relative uncertainty (Poisson + cosmic variance)
		vr = np.sqrt(1.0/(indata.IntNum*indata.Completeness) + meanbias*meanbias*sigma2DM)
		
		text = ""
		text += "\n"
		text += " ------------------------------------------------- \n"
		text += "| Total fractional error on number counts = %.3f | \n" % vr
		text += " ------------------------------------------------- \n"
		
		text += "\n"
		text += "Error Budget: \n"
		text += "Poisson uncertainty (relative): %.3f \n" % (np.sqrt(1.0/(indata.IntNum*indata.Completeness)))
		text += "Cosmic variance (relative): %.3f \n" % (np.sqrt(sigma2DM)*meanbias)
		
		text += "\n"
		text += "Observed counts: %d +/- %d \n" % (int(indata.IntNum*indata.Completeness), int(indata.IntNum*indata.Completeness*vr+0.999))
		
		
		text += " \n"
		text += "Sample properties: \n"
		text += "  Intrinsic Number Density for Galaxies in the Sample [(Mpc/h)^-3] = %.2e \n"  % (beamdata.density*indata.HaloOff)
		text += "  Intrinsic Number Density for Dark Matter Halos Hosts [(Mpc/h)^-3] = %.2e \n" % beamdata.density
		text += "  Minimum DM halo mass [Msun/h] = %.2e \n" % beamdata.minmass
		text += "  Average Bias = %.1f \n" % meanbias
		text += "  Bias @ Minimum DM halo mass = %.1f \n" % beamdata.minbias

		text += " \n"
		text += "Pencil beam properties: \n"
		text += "  Dimension [(Mpc/h)^3] = %.3e x %.3e x %.3e \n" % (beamdata.DX, beamdata.DY, beamdata.DZ)
		text += "  Total volume [(Mpc/h)^3] = %.4e \n" % (beamdata.DX*beamdata.DY*beamdata.DZ)

		text += " \n"
		text += "Input parameters used: \n"
		text += "  *Cosmology* \n"
		text += "    OmegaM = %f \n" % cosmo.OmegaM
		text += "    OmegaL = %f \n" % cosmo.OmegaL
		text += "    h = %f \n"      % cosmo.h
		text += "    Sigma8 = %f \n" % cosmo.sigma8
		if indata.Bias==1:
			text += "    Method = Sheth-Tormen \n"
		else:
			text += "    Method = Press-Schechter \n"
		text += "  *Field of View Pencil Beam* \n"
		text += "    FoV_XSize (arcmin) = %f \n" % indata.SurveyAreaX
		text += "    FoV_YSize (arcmin) = %f \n" % indata.SurveyAreaY
		text += "    Average Redshift = %f \n"   % indata.MeanZ
		text += "    Redshift Width = %f \n"     % indata.Dz
		text += "  *Sample Properties* \n"
		text += "    Intrinsic Number of Objects = %.1f \n" % indata.IntNum
		text += "    Average Halo Occupation = %.1f \n"     % indata.HaloOff
		text += "    Completeness = %.1f \n"                % indata.Completeness

		text += "\n"
		text += "Results obtained using CosmicVarianceCalculator v1.02 \n"
		text += "Developed by Michele Trenti & Massimo Stiavelli \n"
		text += "If you use these results in scientific papers, please refer to:  \n"
		text += "Trenti & Stiavelli (2008), ApJ, 676, 767 \n"
		
		print(text)
