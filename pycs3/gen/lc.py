"""
Module containing all we need to manipulate lightcurves. Could ideally be used by all our curve-shifting algorithms.
"""
import sys
import os
import copy as pythoncopy
import numpy as np
#import util

# For the sort and shuffle stuff :
import random
import operator


class LightCurve:
	"""
	A class to handle light curves of lensed QSO images.

	The actual information is stored into independent "vectors" (1D numpy arrays) : one for time, one for magnitude...

	An exception to this is the properties attribute : a list of dicts, one for each data point, usually all containing the same fields.

	Note that a lightcurve MUST always be chronologically ordered, this allows faster algorithms.
	See methods sort() and validate().
	Importation functions should automatically sort().

	.. note:: There is only one remaining philosophy about handling shifts of lightcurves:  use shifttime() and cie, this is fast and you do not change the actual data (recommended)
	"""

	def __init__(self, telescopename="Telescope", object="Object", plotcolour="crimson"):
		"""
		The plain constructor, binding all the instance variables.
		It gives a small default lightcurve with 5 points, labels, and a mask to play with.

		:param telescopename: string, name of the telescope that took this curve
		:param object: string, name of the astronomical object observed
		:param plotcolour: string, recognized matplotlib colour. Used when plotting the light curve
		"""
		# Some various simple attributes :
		self.telescopename = telescopename
		self.object = object
		self.plotcolour = plotcolour

		# The data itself : here we give some default values for testing purposes :
		self.jds = np.array([1.1, 2.0, 3.2, 4.1, 4.9])  #
		""" 
		The "dates", as floats, in HJD, MHJD, ... these jds should always be in 
		chronological order please (and of course always correspond to the other arrays)."""

		self.mags = np.array([-10.0, -10.1, -10.2, -10.3, -10.2])
		self.magerrs = np.array([0.1, 0.2, 0.1, 0.1, 0.1])

		# Other attributes:
		self.mask = np.array([True, True, False, True, True])
		"""
		A boolean mask, allows to "switch off" some points.
		True means "included", False means "skip this one".
		"""

		self.properties = [{"fwhm": 1.0}, {"fwhm": 1.0}, {"fwhm": 1.0}, {"fwhm": 1.0}, {"fwhm": 1.0}]
		"""
		This is a list of dicts (exactly one per data point) in which you can put whatever you want to.
		You can use them to carry image names, airmasses, sky levels, telescope names etc.
		Of course this slows down a bit things like merging etc.
		"""

		# Other useful settings for plots
		self.ploterrorbars = True
		"""A flag to show or hide errorbars in plots of this lightcurve."""

		# Values of the two obvious possible shifts.
		self.timeshift = 0.0
		"""
		A float giving the shift in time (same unit as jds) that was applied to the lightcurve. 
		This is updated at every shift, so that a lightcurve is always "aware" of its eventual shift. 
		"""

		self.magshift = 0.0
		"""A float giving the shift in magnitude (similar to timeshift)."""

		self.fluxshift = 0.0
		"""
		A float giving the shift in flux (!). As we work with magnitudes, a shift in flux is a bit special.
		Note by the way that a flux shift does not commute with a magnitude shift !
		"""

		# And now the microlensing : by default there is none :
		self.ml = None
		"""Optional, microlensing curve associated to the current light curve"""


	# I explicitly define how str(mylightcurve) will look. This allows a nice "print(mylightcurve)" for instance !
	def __str__(self):
		"""
		We explicitly define how str(mylightcurve) will look. "[telescopename/objet]", or "[telescopename/object](timeshift,magshift)"
		in case the lightcurve is shifted.

		Microlensing "constrains" are shown like |12|, meaning in this case
		two seasons : first one with 1 parameter, second one with two.
		"""
		retstr = "[%s/%s]" % (self.telescopename, self.object)

		if self.timeshift != 0.0 or self.magshift != 0.0 or self.fluxshift != 0.0:
			retstr += "(%.3f,%.3f,%.0f)" % (self.timeshift, self.magshift, self.fluxshift)

		if self.ml != None:
			retstr += "%s" % (self.ml)

		return retstr

	def __len__(self):
		"""
		We define what len(mylightcurve) should return : the number of points.
		I do not exclude masked points -- this is the full length.
		"""
		return len(self.jds)
