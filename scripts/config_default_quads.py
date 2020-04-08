#####################
#  Configuration file
#####################
import os, sys
import pycs
import numpy as np
from module import util_func as ut

#info about the lens :
full_lensname =''
lcs_label = ['A','B','C','D']
delay_labels = ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
#initial guess :
timeshifts = ut.convert_delays2timeshifts([0.,0.,0.])#give the estimated AB delay
magshifts =  [0.,0.,0.,0.]

#general config :
askquestions = False
display = False
max_core = None #None will use all the core available


### OPTIMISATION FUNCTION ###
# select here the optimiser you want to use :
optfctkw = "spl1" #function you used to optimise the curve at the 1st plase (in the script 2a), it should stay at spl1
simoptfctkw = "spl1" #function you want to use to optimise the mock curves, currently support spl1 and regdiff

### SPLINE PARAMETERS ###
knotstep = [25,35,45,55] #give a list of the parameter you want

### REGDIFF PARAMETERS ###
#To use 5 set of parameters pre-selected :
use_preselected_regdiff = True #highly recommended, set to True
preselection_file = 'CHANGE PATH HERE' #'config/preset_regdiff_ECAM.txt'
#You can give your own grid here if use_preselected_regdiff == False :
covkernel = ['matern']  # can be matern, pow_exp or gaussian
pointdensity = [2]
pow = [2.2]
amp = [0.5]
scale = [200.0]
errscale = [25.0]

### RUN PARAMETERS #####
#change here the number of copie and mock curve you want to draw :
# copies
ncopy = 20 #number of copy per pickle
ncopypkls = 25 #number of pickle

# mock
nsim = 20 #number of copy per pickle
nsimpkls = 40 #number of pickle
truetsr = 10.0  # Range of true time delay shifts when drawing the mock curves
tsrand = 10.0  # Random shift of initial condition for each simulated lc in [initcond-tsrand, initcond+tsrand]

## sim
run_on_copies = True
run_on_sims = True


### MICROLENSING ####
mltype = "splml"  # splml or polyml
mllist = [0,1,2,3]  # Which lcs do you want to attach ml to ?
mlname = 'splml'
mlknotsteps = [150,300,450,600]# 0 means no microlensing...
#To force the spacing  :
forcen = False # if true I doesn't use mlknotstep
nmlspl = [0,1,2]  #nb_knot - 1, used only if forcen == True, 0, means no microlensing
#mlbokeps = 88 #  min spacing between ml knots, used only if forcen == True

#polyml parameters :
degree = [0,1,2,3,4] # degree - 1, used only if mltype = "polyml",  0, means no microlensing


###### TWEAK ML #####
#Noise generator for the mocks light curve, script 3a :
tweakml_name = 'PS' #give a name to your tweakml, change the name if you change the type of tweakml, avoid to have _opt_ in your name !
tweakml_type = 'PS_from_residuals' #choose either colored_noise or PS_from_residuals
shotnoise_type = None #Select among [None, "magerrs", "res", "mcres", "sigma"] You should have None for PS_from_residuals

find_tweak_ml_param = True #To let the program find the parameters for you, if false it will use the lines below :
colored_noise_param = [[-2.95,0.001],[-2.95,0.001],[-2.95,0.001],[-2.95,0.001]] #give your beta and sigma parameter for colored noise, used only if find_tweak_ml == False
PS_param_B = [[1.0],[1.0],[1.0],[1.0]] #if you don't want the algorithm fine tune the high cut frequency (given in unit of Nymquist frequency)

#if you chose to optimise the tweakml automatically, you might want to change this
optimiser = 'DIC' # choose between PSO, MCMC or GRID or DIC
n_curve_stat =16# Number of curve to compute the statistics on, (the larger the better but it takes longer... 16 or 32 are good, 8 is still OK) .
n_particles = 50 #this is use only in PSO optimser
n_iter = 80 #number of iteration in PSO or MCMC
mpi = False # if you want to use MPI for the PSO
grid = np.linspace(0.1,1,10) #this is use in the GRID optimiser
max_iter = 15 # this is used in the DIC optimiser, 10 is usually enough.


###### SPLINE MARGINALISATION #########
# Chose the parameters you want to marginalise on for the spline optimiser. Script 4b.
name_marg_spline = 'marginalisation_spline' #choose a name for your marginalisation
tweakml_name_marg_spline = ['PS']
knotstep_marg = knotstep #parameters to marginalise over, give a list or just select the same that you used above to marginalise over all the available parameters
mlknotsteps_marg = mlknotsteps #mlknotsteps or nmlspl if use forcen == True or degree if you use polyml.

###### REGDIFF MARGINALISATION #########
# Chose the parameters you want to marginalise on for the regdiff optimiser. Script 4c.
# We will marginalise over the set of parameters in your preselection_file, you have to create one at this step.
name_marg_regdiff = 'marginalisation_regdiff'
tweakml_name_marg_regdiff = ['PS']
knotstep_marg_regdiff = knotstep #choose the knotstep range you want to marginalise over, by default it is recommanded to take the same as knotstep
mlknotsteps_marg_regdiff = mlknotsteps #mlknotsteps or nmlspl if use forcen == True or degree if you use polyml.

#other parameteres for regdiff and spline marginalisation :
testmode = True # number of bin to use for the mar
sigmathresh = 0.5   #sigma threshold for sigma clipping, 0 is a true marginalisation, choose 1000 to take the most precise.

###### MARGGINALISE SPLINE AND REGDIFF TOGETHER #######
#choose here the marginalisation you want to combine in script 4d, it will also use the sigmathresh:
name_marg_list = ['marginalisation_spline','marginalisation_regdiff']
display_name = ['Free-knot Spline', 'Regression Difference']
new_name_marg = 'marginalisation_final'
sigmathresh_list = [0.5,0.5] #sigmathresh to use for marginalisation_spline and marginalisation_regdiff, it can be different from the sigmathresh used for the new marginalisation
sigmathresh_final = 0.0 #sigma used in the final marginalisation

### Functions definition
if optfctkw == "regdiff" or simoptfctkw == "regdiff":
	from pycs import regdiff

def spl1(lcs, **kwargs):
	# spline = pycs.spl.topopt.opt_rough(lcs, nit=5)
	spline = pycs.spl.topopt.opt_fine(lcs, knotstep=kwargs['kn'], bokeps=kwargs['kn']/3.0, nit=5, stabext=100)
	return spline

def regdiff(lcs, **kwargs):
	return pycs.regdiff.multiopt.opt_ts(lcs, pd=kwargs['pointdensity'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
										amp=kwargs['amp'], scale=kwargs['scale'], errscale=kwargs['errscale'], verbose=True, method="weights")


###### DON'T CHANGE ANYTHING BELOW THAT LINE ######
rawdispersionmethod = lambda lc1, lc2 : pycs.disp.disps.linintnp(lc1, lc2, interpdist=interpdist)
dispersionmethod = lambda lc1, lc2 : pycs.disp.disps.symmetrize(lc1, lc2, rawdispersionmethod)
def disp(lcs):
	return pycs.disp.topopt.opt_full(lcs, rawdispersionmethod, nit=3, verbose=True)

def attachml(lcs, ml):
	if ml == 0 : #I do nothing if there is no microlensing to attach.
		return
	lcmls = [lcs[ind] for ind in mllist]
	mlvec = [ml for ind in mllist] # this is either the number of knot, either a knot step depending if forcen is True or False
	if mltype == 'splml':
		if forcen:
			for lcml, nml in zip(lcmls, mlvec):
				curve_length = lcml.jds[-1] - lcml.jds[0]
				mlbokeps = np.floor(curve_length / nml)

				if nml == 1 : #spline cannot have 0 internal knot, then we use a polynome of degree 2 to represents a spline with only two external knot
					pycs.gen.polyml.addtolc(lcml,  nparams=3 )
				else :
					pycs.gen.splml.addtolc(lcml, n=nml, bokeps=mlbokeps)
		else:
			for lcml, mlknotstep in zip(lcmls, mlvec):
				mlbokeps_ad = mlknotstep / 3.0   #maybe change this
				pycs.gen.splml.addtolc(lcml, knotstep=mlknotstep, bokeps=mlbokeps_ad)

	# polynomial microlensing
	nparams = [ml for ind in mllist]
	if mltype == 'polyml':
		for ind, lcml in enumerate(lcmls):
			pycs.gen.polyml.addtolc(lcml, nparams=nparams[ind], autoseasonsgap = 60.0)

if optfctkw == "spl1":
	optfct = spl1
	splparamskw = ["ks%i" %knotstep[i] for i in range(len(knotstep))]

if optfctkw == "regdiff": # not used, small haxx to be able to execute 2 to check and 3 using the spl1 drawing
	optfct = regdiff
	splparamskw = "ks%i" % knotstep

if simoptfctkw == "spl1":
	simoptfct = spl1

if simoptfctkw == "regdiff":
	simoptfct = regdiff
	if use_preselected_regdiff :
		regdiffparamskw = ut.read_preselected_regdiffparamskw(preselection_file)
	else :
		regdiffparamskw = ut.generate_regdiffparamskw(pointdensity,covkernel, pow, amp, scale, errscale)


if mltype == "splml" :
	if forcen == False :
		combkw = [["%s_ks%i_%s_ksml_%i" %(optfctkw, knotstep[i], mlname, mlknotsteps[j]) for j in range(len(mlknotsteps))]for i in range(len(knotstep))]
	else :
		combkw = [["%s_ks%i_%s_nmlspl_%i" % (optfctkw, knotstep[i], mlname, nmlspl[j]) for j in range(len(nmlspl))]for i in range(len(knotstep))]
elif mltype == "polyml":
	combkw = [["%s_ks%i_%s_deg_%i" % (optfctkw, knotstep[i], mlname, degree[j]) for j in range(len(degree))] for i in range(len(knotstep))]
else :
	raise RuntimeError('I dont know your microlensing type. Choose "polyml" or "spml".')
combkw = np.asarray(combkw)

simset_copy = "copies_n%i" % (int(ncopy * ncopypkls))
simset_mock = "mocks_n%it%i_%s" % (int(nsim * nsimpkls), truetsr,tweakml_name)

if simoptfctkw == "regdiff":
	if use_preselected_regdiff :
		kwargs_optimiser_simoptfct = ut.get_keyword_regdiff_from_file(preselection_file)
		optset = [simoptfctkw + regdiffparamskw[i] + 't' + str(int(tsrand)) for i in range(len(regdiffparamskw))]
	else :
		kwargs_optimiser_simoptfct = ut.get_keyword_regdiff(pointdensity, covkernel, pow, amp, scale, errscale)
		optset = [simoptfctkw + regdiffparamskw[i] + 't' + str(int(tsrand)) for i in range(len(regdiffparamskw))]
elif simoptfctkw == 'spl1':
	optset = [simoptfctkw + 't' + str(int(tsrand))]
else :
	print('Error : I dont recognize your simoptfctkw, please use regdiff or spl1')
	sys.exit()
