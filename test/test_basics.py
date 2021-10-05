import unittest
import radialdf
import numpy as np
import plotly.graph_objects as go

class TestRDF(unittest.TestCase):
    def test_box(self):
        # Generate 10000 random particles with 3 coordinates between 0 and 10
        side = 10
        resolution = 0.2
        particles = np.random.rand(1000, 3) * side
        # Define a volume from 0 to 100 on 3 axes
        box = radialdf.Box(side=side)
        g = radialdf.volume_rdf(box, particles, 4, resolution)
        ##self.assertAlmostEqual(4, len(g), "This should have been 4")
        print("RESULT:", g)
        go.Figure(go.Scatter(x=[i * resolution for i in range(int(side // resolution))], y=g)).show()
    
    def test_lala(self):
        pass