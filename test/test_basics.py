import unittest
import radialdf
import numpy as np
import plotly.graph_objects as go

class TestRDF(unittest.TestCase):
    def test_box(self):
        # Generate 1000 random particles with 3 coordinates between 0 and 10
        side = 10
        resolution = 0.2
        particles = np.random.rand(1000, 3) * side
        # Define a volume from 0 to 100 on 3 axes
        box = radialdf.Box(side=side)
        g = radialdf.volume_rdf(box, particles, 4, resolution)
        ##self.assertAlmostEqual(4, len(g), "This should have been 4")
        print("RESULT:", g)
        #go.Figure(go.Scatter(x=[i * resolution for i in range(int(side // resolution))], y=g)).show()
    

    def test_sphere(self): 
        # Generate 1000 random particles in a sphere with radius 100
        radius = 100
        num_particles = 1000
        particles = []
        for i in range(num_particles):
            u, v = np.random.rand(2,1)
            r = radius / 2 * np.random.random()
            theta = 2.0 * u * np.pi 
            phi = np.arccos(2.0 * v - 1.0)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            particles.append((x,y,z))

        resolution = 0.2
        sphere = radialdf.Sphere(radius=radius)
        particles = np.array(particles).reshape(num_particles, 3)
        g = radialdf.volume_rdf(sphere, np.array(particles), 5, resolution)
        ##self.assertAlmostEqual(4, len(g), "This should have been 4")
        print("RESULT:", g)
        go.Figure(go.Scatter(x=[i * resolution for i in range(int(radius // resolution))], y=g)).show()


    def test_lala(self):
        pass