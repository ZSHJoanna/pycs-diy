Download & Installation
=======================


Dependencies
------------

PyCS3 is developed using python 3.7 and might be fine with older versions.

It requires ``numpy``, ``scipy`` and ``matplotlib``.
Those are all you need to run the free-knot splines.

The regression difference technique have furter dependencies:

* `scikit-learn <http://scikit-learn.org>`_


Download
--------

Get the latest PyCS by cloning it from `GitLab <https://gitlab.com/vbonvin/PyCS3>`_::

	git clone https://gitlab.com/vbonvin/PyCS3


If for some reason you want the exact version of PyCS that was used in the original paper, you can still find it here::

	svn export https://svn.epfl.ch/svn/pycs/tags/PyCS-1.0 ./PyCS-1.0
	


Installation
------------

If you  want to update or tweak the sources, we suggest to just add your cloned repository to your ``PYTHONPATH`` or type :

::

    python setup.py develop

If you don't plan to tweak the code, you can also simply

::

	python setup.py install

or maybe

::

	python setup.py install --user

... if you don't have write access to the global site-packages directory of your machine.