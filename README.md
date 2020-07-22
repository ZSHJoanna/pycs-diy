# PyCS3


[![Documentation Status](https://vbonvin.gitlab.io/PyCS3/doc_status.svg)](https://vbonvin.gitlab.io/PyCS3/)
[![pipeline status](https://gitlab.com/vbonvin/PyCS3/badges/master/pipeline.svg)](https://gitlab.com/vbonvin/PyCS3/commits/master)
[![coverage report](https://gitlab.com/vbonvin/PyCS3/badges/master/coverage.svg)](https://vbonvin.gitlab.io/PyCS3/coverage/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


PyCS3 is a software toolbox to estimate time delays between multiple images of gravitationally lensed quasars, developed within the [COSMOGRAIL](http://www.cosmograil.org) collaboration. This is an update of [PyCS](https://github.com/COSMOGRAIL/PyCS), that is no longer maintained. 


Proceed to the [documentation](https://vbonvin.gitlab.io/PyCS3/) to get further information. In case of any questions, feel free to open an issue here on GitLab.

## Installation 

    git clone https://gitlab.com/vbonvin/PyCS3.git
    cd PyCS3 
    python setup.py install

or if you prefer to install it locally : 

    python setup.py install --user 
    
## Requirements 

PyCS3 requires the following standard python packages : 
* `numpy`
* `scipy`
* `matplotlib`

If you want to use the regdiff optimiser, you will also need : 
* `scikit-learn`
    
## Example Notebooks and Documentation
The full documentation can be found [here](https://vbonvin.gitlab.io/PyCS3/). 

Example notebooks are located in the [notebook](https://gitlab.com/vbonvin/PyCS3/-/tree/master/notebook) folder : 
* [Importing, exporting and displaying light curves](https://gitlab.com/vbonvin/PyCS3/-/blob/master/notebook/Import_export_and_display.ipynb)
* [Measuring time delays with regdiff and the splines](https://gitlab.com/vbonvin/PyCS3/-/blob/master/notebook/Measuring%20time%20delays%20with%20spline%20and%20regdiff.ipynb)
* [Estimating uncertainties with PyCS3](https://gitlab.com/vbonvin/PyCS3/-/blob/master/notebook/Uncertainties%20estimation.ipynb)

## Attribution

If you use this code, please cite [the papers](https://vbonvin.gitlab.io/PyCS3/citing.html) indicated in the documentation.

## License
PyCS3 is a free software ; you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation ; either version 3 
of the License, or (at your option) any later version.

PyCS3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details ([LICENSE.txt](LICENSE)).

