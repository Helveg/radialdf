"""
Radial Distribution Function module
"""
from scipy.spatial import KDTree
import numpy as np

__version__ = "0.0.1"

def inner_rdf(boundary, particles, r, dr):
    """
    Computes the radial distribution function, showing the average density of other
    particles around each particle sampled in concentric shells of thickness ``dr`` within
    a maximum radius of ``r``. This function avoids the outer ``r`` edges of each
    dimension as it currently does not deal with border effects. It will calculate the
    RDF based only on those particles located at least ``r`` away from all borders.

    The ``particles`` argument should have an (n, k) shape where n is the amount of
    particles and k the number of dimensions.

    The ``boundary`` argument should have shape (k, 2) where each row is the mimimum and
    maximum of a dimension of the volume within which particles should have been placed.
    An example for 3d-data would be [[0, 100], [0, 200], [0, 100]] for a 100x200x100 box.

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
    tree = KDTree(particles)
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

def _within(particles, boundary):
    within = None
    for _dim, (low, high) in enumerate(boundary):
        dim = particles[:, _dim]
        mask = (dim >= low) & (dim <= high)
        if within is not None:
            mask = mask & within
        within = mask
    return np.nonzero(within)[0]

def rdf(boundary, particles, r, dr):
    raise NotImplementedError("The RDF dealing with border effects is not implemented yet, use `inner_rdf` instead, which excludes the border regions.")
