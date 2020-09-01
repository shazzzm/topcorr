.. TopCorr documentation master file, created by
   sphinx-quickstart on Fri May 22 17:57:08 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TopCorr's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

TopCorr is a small python library for constructing filtered correlation networks. Currently implemented so far:

* MST
* PMFG
* TMFG
* Thresholding
* Dependency Network
* k-Nearest Neighbours Network
* Partial Correlation
* Affinity Matrix
* Average Linkage MST
* Forest of MSTs
* Detrended Cross Correlation Analysis


As a rule of thumb, if the network returned from a method is sparse, it will be a networkx graph. If it is a dense network, it will be in the form of a correlation matrix.

Methods
==================
.. automodule:: topcorr
	   :members:

.. automodule:: topcorr.topcorr
	   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
