# From https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

import numpy as np
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools

#  Example
# m, n, d = 3.5e4, 3e3, 3
# # make sure no new grid point is extrapolated
# bounding_cube = np.array(list(itertools.product([0, 1], repeat=d)))
# xyz = np.vstack((bounding_cube,
#                  np.random.rand(m - len(bounding_cube), d)))
# f = np.random.rand(m)
# g = np.random.rand(m)
# uvw = np.random.rand(n, d)
#
# In [2]: vtx, wts = interp_weights(xyz, uvw)
#
# In [3]: np.allclose(interpolate(f, vtx, wts), spint.griddata(xyz, f, uvw))
# Out[3]: True

# The code below replaces
#     tri = Delaunay(data_loc)
#     interpolator = LinearNDInterpolator(tri, interp_data)
#     interp_field = interpolator(interp_loc)
#
# with
#
#     interpolator = PreWeightedLinearNDInterpolator(data_loc, interp_loc)
#     interp_field = interpolator(interp_data)



def interp_weights(xyz, uvw, d):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

class PreWeightedLinearNDInterpolator(object):
    def __init__(self, data_loc, interp_loc):
        self.data_tri, self.weights = interp_weights(data_loc,
            interp_loc, data_loc.shape[1])

    def __call__(self, interp_data):
        return interpolate(interp_data, self.data_tri, self.weights)

