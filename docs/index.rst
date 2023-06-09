.. PyCS3 documentation master file, created by
   sphinx-quickstart on Fri Aug 10 11:19:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyCS3's documentation!
=================================

.. image:: _static/cover_image_1.png
	:align: center


About
-----

PyCS3 is a software toolbox to estimate time delays between multiple images of strongly lensed quasars, from resolved light curves such as obtained by the `COSMOGRAIL <http://www.cosmograil.org/>`_ monitoring program. It comes in the form of a python package, and heavily depends on ``numpy``, ``scipy``, and ``matplotlib`` for its core functionality. The `repository is on GitLab <https://gitlab.com/cosmograil/PyCS3>`_.

To measure time delays with ``pycs3``, you'll typically write a script calling some high-level functions provided by the package. PyCS3 allows you to compare different point estimators (including your own), without much code integration. You can follow the `example notebooks <https://gitlab.com/cosmograil/PyCS3/-/tree/master/notebook>`_ to learn how to use the core functionnality of PyCS3.

If you have already read our :doc:`papers<citing>`, you might want to proceed with :doc:`installation`, or the :doc:`Tutorial <tutorial/index>`.

.. warning:: Please read this :doc:`important warning about using PyCS3<warning>`.

Questions ?
-----------

Feel free to post an `issue on GitLab <https://gitlab.com/cosmograil/PyCS3/-/issues>`_, or to contact the code authors `Martin Millon <http://people.epfl.ch/martin.millon>`_, `Malte Tewes <https://astro.uni-bonn.de/~mtewes>`_ and `Vivien Bonvin <http://people.epfl.ch/vivien.bonvin>`_.


Contents
--------


.. toctree::
   :maxdepth: 2

	A word of warning <warning>
	Download & Installation <installation>
	Tutorial<tutorial/index>
    Intra-dependency chart <chart>
	Citing <citing>
    Published work with PyCS <publication>
	Autogenerated Full API <apidoc/pycs3>


Last build of this documentation : |today|.


