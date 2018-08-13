"""
Testing script, v1
"""

import os
import sys

# add the root directory to sys.path, so pycs3 can be imported without installation
path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), '../')
sys.path.append(path)

import pycs3


lc = pycs3.gen.lc.LightCurve()
print(lc)

assert(2==2)