import os
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func
import pycs3.regdiff.multiopt as multiopt
import pycs3.regdiff.rslc as rslc

path = '/Users/martin/Desktop/modules/PyCS3/tests/'
outpath = os.path.join(path, "output")
rdbfile = os.path.join(path, "data", "trialcurves.txt")
lcs = [
    lc_func.rdbimport(rdbfile, object='A', magcolname='mag_A', magerrcolname='magerr_A',
                      telescopename="Trial"),
    lc_func.rdbimport(rdbfile, object='B', magcolname='mag_B', magerrcolname='magerr_B',
                      telescopename="Trial"),
    lc_func.rdbimport(rdbfile, object='C', magcolname='mag_C', magerrcolname='magerr_C',
                      telescopename="Trial"),
    lc_func.rdbimport(rdbfile, object='D', magcolname='mag_D', magerrcolname='magerr_D',
                      telescopename="Trial")
    ]

mrg.colourise(lcs)
for lc in lcs :
    print(lc.timeshift)

lc_func.settimeshifts(lcs,shifts=[0, -5, -20, -60],includefirst=True)

# myrslcs, error_fct = multiopt.opt_ts(lcs, pd=2, covkernel='matern',
#         pow=1.5, amp=1., scale=200., errscale=1., verbose=True, method="weights") # good set for mattern

# myrslcs, error_fct = multiopt.opt_ts(lcs, pd=2, covkernel='matern',
#         pow=2.5, amp=1.0, scale=200., errscale=1., verbose=True, method="weights") # good set for mattern
#

myrslcs, error_fct = multiopt.opt_ts(lcs, pd=2, covkernel='RBF',
                pow=1.5, amp=0.5, scale=200., errscale=1., verbose=True, method="weights") # good set for Radial-Basis Function
# myrslcs, error_fct = multiopt.opt_ts(lcs, pd=2, covkernel='RBF',
#         pow=1.5, amp=0.5, scale=200., errscale=1., verbose=True, method="weights") # good set for RBF , pow is not used
#
# myrslcs, error_fct = multiopt.opt_ts(lcs, pd=2, covkernel='RatQuad',
#         pow=1., amp=0.5, scale=200., errscale=1., verbose=True, method="weights") # good set for mattern

lc_func.display(lcs, myrslcs, filename='screen')
for lc in myrslcs:
    print(lc.timeshift)