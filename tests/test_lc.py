"""
Testing script, v1
"""

import os
import sys

# add the root directory to sys.path, so pycs3 can be imported without installation
path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), '../')
sys.path.append(path)

from pycs3.gen.lc_func import rdbimport
from pycs3.gen.mrg import colourise

rdbfile = "data/trialcurves.txt"
lcs = [
    rdbimport(rdbfile, object='A', magcolname='mag_A', magerrcolname='magerr_A',
                                telescopename="Trial"),
    rdbimport(rdbfile, object='B', magcolname='mag_B', magerrcolname='magerr_B',
                                telescopename="Trial"),
    rdbimport(rdbfile, object='C', magcolname='mag_C', magerrcolname='magerr_C',
                                telescopename="Trial"),
    rdbimport(rdbfile, object='D', magcolname='mag_D', magerrcolname='magerr_D',
                                telescopename="Trial")
]

colourise(lcs)  # Gives each curve a different colour.
#Check the length of the light curve
assert (len(lcs[0]) == 192)

lc = lcs[0]
print(lc)
stats=lc.samplingstats()
properties = lc.commonproperties()
properties_all = lc.commonproperties(notonlycommon=True)
print(stats)


