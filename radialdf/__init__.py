from operator import ne
import numpy as np 
from scipy.spatial import KDTree
import functools
import plotly.graph_objects as go 

PLOTTING = False 

class Volume():
    def __init__(self, boundary, resolution): 
        self.boundary = boundary
        self.resolution = resolution 
    
    @functools.lru_cache()  
    # NOTE: lru_cache without parentheses works with Python > 3.8
    def _get_grid(self):
        grid = np.mgrid[tuple(slice(l, h, self.resolution) for l, h in self.boundary)]
        grid = np.column_stack(tuple(x.reshape(-1) for x in grid))
        shape = grid[[self.hit_function(x) for x in grid]]
        return shape, KDTree(shape)

    def query_grid(self, points, r):
        grid, grid_tree = self._get_grid()
        return [grid[p] for p in grid_tree.query_ball_point(points, r=r)]
        #return grid
    
    def hit_function(self, x):
        return True



def volume_rdf(volume, particles, r, dr):
    """
    Computes the radial distribution function, showing the average density of other
    particles around each particle sampled in concentric shells of thickness ``dr`` within
    a maximum radius of ``r``. This function stays a distance of ``r`` away from all
    borders as it currently does not deal with border effects. It will calculate the RDF
    only for those particles located at least ``r`` away from each border.

    The ``particles`` argument should have an (n, k) shape where n is the amount of
    particles and k the number of dimensions.

    The ``boundary`` argument should have shape (k, 2) where each row is the mimimum and
    maximum of a dimension of the volume within which particles should have been placed.
    An example for 3d-data would be ``[[0, 100], [0, 200], [0, 100]]`` for a 100x200x100
    box.

    :param boundary: The limits for each dimension of the particles. Used to normalize to volume and exclude boundary particles.
    :type boundary: np.ndarray-like with shape (k, 2)
    :param particles: The particles to investigate.
    :type particles: np.ndarray-like with shape (n, k)
    :param r: The maximum search radius, and the boundary exclusion size
    :type r: float
    :param dr: Resolution of the RDF. Thickness of the radial shells whose particle density is the basis of the RDF.
    :type dr: float
    """

    tree = KDTree(particles)
    # Query the tree for the neighbours within radius r of all the reference particles.
    all_neighbours = tree.query_ball_point(particles, r=r)
    grid_points = volume.query_grid(particles, r=r)
    print(grid_points)

    if PLOTTING:
        fig = go.Figure(data=[go.Scatter3d(x=particles[:,0],y=particles[:,1],z=particles[:,2],
                        mode='markers')])
        for i in grid_points:
            fig.add_trace(go.Scatter3d(x=i[:,0],y=i[:,1],z=i[:,2],
                        mode='markers', marker=dict(size=1)))
        fig.show()


    # Global density
    grid, grid_tree = volume._get_grid()
    density = len(particles) / len(grid)
    del grid, grid_tree

    radii = np.arange(0, r, dr)
    bin_densities = np.zeros(len(radii))
    for origin, neighbors, local_grid in zip(particles, all_neighbours, grid_points):
        print("Computing a new particle's density")
        index = np.where(particles == origin)[0][0]
        neighbors.remove(int(index))
        pos = np.abs(particles[neighbors] - origin)
        grid_pos = np.abs(local_grid - origin)
        prev = [0, 0]
        bins = [[], []]

        for r in radii:
            for t, coll in enumerate((pos, local_grid)):    
                in_radius = sum(np.sqrt(np.sum(coll ** 2, axis=1)) < r)
                in_shell = in_radius - prev[t]
                prev[t] = in_shell
                bins[t].append(in_radius) 
            
            # TODO: add a way to filter 0 on denominator
            bin_densities += np.array(bins[0]) / np.array(bins[1])

    print(bin_densities)
    
    return bin_densities / len(particles) / density 


# Generate 10000 random particles with 3 coordinates between 0 and 100
side = 10
particles = np.random.rand(50, 3) * side
# Define a volume from 0 to 100 on 3 axes
box = [[0, side]] * 3
# Check the radial distribution, which should be pretty boring and flat
v = Volume(box, resolution=0.2)
# NOTE: user-defined atlas / hit function 
#v.hit_function = lambda x: x[0] < 50
g = volume_rdf(v, particles, 3, 0.2)
print(g)
#go.Figure(go.Scatter(x=[i * 0.2 for i in range(21)], y=g)).show()