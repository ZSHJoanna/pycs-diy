Tips and Tricks, code snippets
==============================


.. contents::


Which PyCS am I using ?
-----------------------

If you have several copies, from SVN or installed...
::

	import pycs3
	print(pycs3.__file__)
	


"Faking" PyCS3 delay or error measurements
-----------------------------------------

This is useful if you want to include measurements from non-pycs techniques on the typical pycs3 plots
::

	# To fake a PyCS3 delay measurement :
	data = [{"label":"AB", "mean":-118.6, "med":0.0, "std":0.0}]

	# To fake a PyCS3 errorbar :
	"""
	data = [{
	"label":"AB",
	"sys":3.0,
	"ran":4.0,
	"tot":5.0,
	"bias":-3.0
	}]
	"""
	outname = "sims_copies_runresults_rathna_May07_delays.pkl"
	dc = pycs3.sim.plot.delaycontainer(data = data, name = "Difference-smoothing technique", plotcolour = "darkorange", objects=["A", "B"])
	pycs3.gen.util.writepickle(dc, outname)


Building scrolling plots for long curves
----------------------------------------

::

	# Animated plot for talk :

	startjd = 52900.0
	width = 1000.0
	endjd = 55800.0
	n = 1000
	
	for i in range(n):
		
		a = startjd + i* (endjd - width - startjd)/(n-1)
		b = a + width
		
		filename = "mov/%i.png" % (i)
		pycs3.gen.lc_func.display(lcs, nicefont=True, showdelays=False, showlegend=False, showdates=True, showgrid=True, magrange=(4.3, 0), jdrange=(a, b), filename=filename)
	

And then use ffmpeg (or any other similar tool) to turn this into a movie.

Tweaking magnitudes for individual seasons
------------------------------------------

For a lightcurve ``l``, ``l.mags`` is just a numpy array.
To *lower* the third season by 0.03 mags :
::
	
	seasons = pycs3.gen.sea.autofactory(l)
	l.mags[seasons[2].indices] += 0.03
	



Playing with custom properties
------------------------------

You can perfectly create your own properties. It's just a list of dicts ...
::
	
	for i in range(len(l)):
		l.properties[i]["my_new_prop"] = "brocoli"
		
	# To see what properties a curve has :
	print(l.longinfo())

"Common" properties are properties that all points of the curve have (this is usually the case). Only those "common" properties can be exported as columns in rdb files, for instance.


Splitting a curve by properties
-------------------------------

::
	
	def splitbyprop(l, prop = "telescope"):
		"""
		kills mask ...
		"""
		
		vals = sorted(list(set([l.properties[i][prop] for i in range(len(l))])))
		
		out = []
		for val in vals:
			lcp = l.copy()
			lcp.mask = np.array([l.properties[i][prop] == val for i in range(len(l))])
			lcp.cutmask()
			lcp.telescopename = val
			out.append(lcp)
			
		#pycs3.gen.mrg.colourise(out)
		return out




Correcting for flux sharing
---------------------------

March 2012, only implemented for the spline method. Simple code works well, but quick tests on simulated data (HE2149) show degeneracies.
Need complete tests on simulated data with a little flux sharing, to see if it reduces systematic error.

::

	# draw fake curves :
	flcs = pycs3.sim.draw.draw(lcs, spline, shotnoise="none", keepshifts=False)
	pycs3.sim.draw.shareflux(flcs[0], flcs[1], frac=0.02)
	pycs3.gen.lc_func.display(flcs)

	# then run pycs3.spl.topopt.opt_fine, it has the option "redistribfluxes"
