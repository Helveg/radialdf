import numpy as np 
from scipy.spatial import cKDTree
import functools
import plotly.graph_objects as go 
import math
import itertools

PLOTTING = False
_pi = math.pi

try:
    functools.cache
except AttributeError:
    functools.cache = functools.lru_cache(None)

def _pairs(iterable):
    "s -> (0, s0), (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    a0 = next(b, None)
    yield 0, a0
    yield from zip(a, b)

def _sphere_volume(r):
    return 4 / 3 * _pi * r ** 3

def _shell_volume(r_pair):
    return _sphere_volume(r_pair[1]) - _sphere_volume(r_pair[0])

class Volume():
    def __init__(self, boundary, resolution): 
        self.boundary = boundary
        self.resolution = resolution

    def get_shells(self, r, dr, points_per_shell):
        # TODO: use particle_per_shell quantities and not hard-coded values
        radii = np.arange(dr / 2, r, dr)
        particles_pos = []
        d_phi = np.linspace(0, _pi, 10)
        d_theta = np.linspace(0, 2*_pi, 10)
        shells = [[] for _ in range(len(radii))]
        for shell, radius in enumerate(radii):
            for n in d_theta:
                for m in d_phi:
                    x = radius * np.sin(m) * np.cos(n)
                    y = radius * np.sin(m) * np.sin(n)
                    z = radius * np.cos(m)
                    shells[shell].append([x, y, z])

        shell_pairs = _pairs(iter(radii))
        shell_volumes = np.fromiter(map(_shell_volume, shell_pairs), dtype=float)

        return [np.array(shell) for shell in shells], shell_volumes
    
    def hit_function(self, x):
        raise NotImplementedError(
            "Volumes should implement a `hit_function`."
            + " Given a point, return whether it is in the volume or not."
        )
    
    def _get_hit_function(self):
        return np.vectorize(self.hit_function, signature="(d)->()")
    
    def total_volume(self):
        raise NotImplementedError(
            "Volumes should implement a `total_volume`"
            + " which returns the global count of all the points in the grid."
        )


class Box(Volume):
    def __init__(self, side, resolution=0.2):
        self._side = side
        self._resolution = resolution
        super().__init__([[0, side]] * 3, resolution)

    def hit_function(self, pos):
        return np.all((pos >= 0) & (pos <= self._side))
    
    def total_volume(self):
        return self._side ** 3


class Sphere(Volume):
    def __init__(self, radius, resolution=0.2):
        self._resolution = resolution
        self.radius = radius
        super().__init__([[-radius, radius]] * 3, resolution)
    
    def hit_function(self, x):
        return np.sum(x ** 2)  < self.radius ** 2

    def total_volume(self):
        return _sphere_volume(self.radius)  


def inner_rdf(boundary, particles, r, dr):
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
    # We will transform the boundary to a numpy array to sanitize input and convenience,
    # but not for `particles` as this might copy this possibly very large array. Instead,
    # we expect `particles` to be given in an appropriate ndarray-like type already.
    boundary = np.array(boundary)
    if boundary.shape[0] != particles.shape[1]:
        raise Exception("The given boundaries do not match the dimensionality of the given particles.")
    search_box = boundary.copy()
    # Shrink the search box to within the maximum search radius to avoid border effects
    # This does mean that the radial distribution of particles in this border region is
    # not examined. See #1 for future implementation of the overlap library to fix this.
    search_box[:, 0] += r
    search_box[:, 1] -= r
    # Find all the reference particles that fall within the shrunken bounds and won't
    ref = _within(particles, search_box)
    tree = cKDTree(particles)
    # Query the tree for the neighbours within radius r of all the reference particles.
    all_neighbours = tree.query_ball_point(particles[ref], r=r)
    # Create an iterator that turns the reference id into the particle id and the
    # neighbour list into a neighbour np.ndarray
    nbr_iter = map(lambda x: (ref[x[0]], np.array(x[1])), enumerate(all_neighbours))
    # A local function that turns neighbour ids into distances and excludes the query
    # particle with id `me`, using the `particles` of this function call scope.
    def dist_excl_self(me, my_nbr):
        offsets = particles[my_nbr[my_nbr != me]] - particles[me]
        return np.sqrt(np.sum(offsets ** 2, axis=1))
    # Concatenate all the distances of all the neighbours of all the reference points
    # together
    all_distances = np.concatenate(tuple(dist_excl_self(me, my_nbr) for me, my_nbr in nbr_iter))
    # Bin them in `dr` sized bins
    radial_bins = np.bincount(np.floor(all_distances / dr).astype(int))
    # Normalize the result to the amount of reference points were tested
    trials = len(ref)
    # Normalize the bins to the volume of the shell that corresponds to the radial bin
    radii = np.arange(0, len(radial_bins) * dr, dr)
    inner_volumes = 4 / 3 * np.pi * radii ** 3
    outer_volumes = 4 / 3 * np.pi * (radii + dr) ** 3
    bin_volumes = outer_volumes - inner_volumes
    # Normalize the result to the density of the particles
    density = len(particles) / np.product(np.abs(np.diff(boundary, axis=1)))
    # Return the normalized radial distribution
    return radial_bins / trials / bin_volumes / density


def volume_rdf(volume, particles, r, dr, shell_points=100):
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
    :param shell_points: Volume grid points to investigate per shell.
    :type shell_points: int
    """
    tree = cKDTree(particles)
    # Query the tree for the neighbours within radius r of all the reference particles.
    print("Querying ball points")
    all_neighbours = tree.query_ball_point(particles, r=r)
    print("Ball points retrieved")
    density = len(particles) / volume.total_volume()

    radii = np.concatenate((np.arange(dr, r, dr), [r]))
    bin_densities = np.zeros(len(radii))
    shells, shell_volumes = volume.get_shells(r, dr, shell_points)
    hit_func = volume._get_hit_function()
    
    print("Shells retrieved: len {}".format(len(shells)))

    print("Iterating over particles")
    for index, (origin, neighbours) in enumerate(zip(particles, all_neighbours)):
        print(f"Computing a new particle's density ({index}, {origin}) with {len(neighbours) - 1} neighbours.")
        neighbours_pos = particles[neighbours]
        prev_n = 0
        bins = np.empty(len(radii))
        for id, sr, shell, shell_volume in zip(itertools.count(), radii, shells, shell_volumes):
            # Count how much of the volume around `origin` is inside the user volume.
            occupancy = np.count_nonzero(hit_func(shell + origin)) / shell_points
            # Multiply by the volume of the shell we are investigating, for normalization.
            shell_v = shell_volume * occupancy
            # Count how many of the points are within `shell` distance from `origin`.
            n = np.count_nonzero(np.sum((neighbours_pos - origin) ** 2, axis=1) < sr ** 2)
            # Don't count points that are members of the smaller (previous) shells.
            shell_n = n - prev_n
            if id == 0:
                # Exclude the query point itself, which will be found in the first bin.
                shell_n -= 1
            prev_n = n
            assert shell_n >= 0, "negative neighbours"
            assert shell_volume > 0, "negative shell volume"
            assert occupancy > 0 and occupancy <= 1, "weird occupancy"
            bins[id] = shell_n / shell_v

        bin_densities += bins

    # Normalize the accumulated results by amount of points investigated, and by global density.
    return np.array(bin_densities) / len(particles) / density


def _within(particles, boundary):
    within = None
    for _dim, (low, high) in enumerate(boundary):
        dim = particles[:, _dim]
        mask = (dim >= low) & (dim <= high)
        if within is not None:
            mask = mask & within
        within = mask
    return np.nonzero(within)[0]