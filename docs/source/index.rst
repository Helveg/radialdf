.. RadialDF documentation master file, created by
   sphinx-quickstart on Fri Jan  1 00:11:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RadialDF's documentation!
====================================

This package provides the radial distribution function to analyze the radial density of
particles around other particles. The package provides a single function :func:`.inner_rdf` that
calculates the RDF but excludes the border regions (as they would require n-dimensional
intersection of shapes, which isn't computationally feasible).

A future version, if my short attention span permits it, will provide an ``rdf`` function
that handles the 2D and 3D cases with the inclusion of the border regions.

Example
-------

.. code-block:: python

   from radialdf import inner_rdf
   import numpy as np
   import plotly.graph_objs as go

   # Generate 10000 random particles with 3 coordinates between 0 and 100
   particles = np.random.rand(10000, 3) * 100
   # Define a volume from 0 to 100 on 3 axes
   box = [[0, 100]] * 3
   # Check the radial distribution, which should be pretty boring and flat
   g = inner_rdf(box, particles, 20, 0.2)
   go.Figure(go.Scatter(x=[i * 0.2 for i in range(21)], y=g)).show()


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   radialdf

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
