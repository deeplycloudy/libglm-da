"""
Read GLM grids, which have various extent density-type products, perform
parallax correction, and subset to point lat lon "observations" that subsample
the GLM observations for use in data assimilation. This approach retains the
event-level spatial extent information from each flash. Think of it as point
sampling of the observed GLM image for cross-comparison of with predicted flash
extent so that the model state can be adjusted through the incremental
information provided by GLM observations.


Eric Bruning, 4 Nov 2020
"""
import sys
import xarray as xr
import numpy as np

from geometry import get_glm_earth_geometry

def glm_locations(glm, ltgellipsever=1):
    """ Given a glm gridded data file, add the lat, lon locations of each pixel
    in the image, corrected to their lightning ellipse position.

    Returns a new xarray dataset with 2D variables matching others (e.g.,
    flash extent density)
    - lon_ltg_ellps, lat_ltg_ellps: the locations
    - in_fov: boolean, True where pixel is within the field of view


    """

    # Convert fixed grid satellitie coords to parallax-corrected lat-lon
    nadir = glm.nominal_satellite_subpoint_lon.data
    # x, y are the fixed grid 2D coord arrays (from meshgrid)
    # X, Y, Z are the ECEF coords of each pixel intersected at the ltg ellipsoid
    # lon_ltg, lat_ltg are the parallax-corrected lon lat at the earth's surface
    # below the lightning position on the lightning ellipsoid.
    # outside_glm_full_disk is a boolean mask for the positions that GLM can't
    #   observe.
    # All of the above are 2D arrays corrsponding the center positions of the 2
    # km fixed grid pixels in the GLM gridded products.
    ((x, y),
     (X, Y, Z),
     (lon_ltg, lat_ltg, alt_ltg),
     outside_glm_full_disk,
     ) = get_glm_earth_geometry(glm, nadir, ltgellipsever)

    img_dim = glm.flash_extent_density.dims

    good = np.isfinite(lon_ltg) & np.isfinite(lat_ltg) & ~outside_glm_full_disk

    glm['lon_ltg_ellps'] = xr.DataArray(lon_ltg, dims=img_dim)
    glm['lat_ltg_ellps'] = xr.DataArray(lat_ltg, dims=img_dim)
    glm['in_fov'] = xr.DataArray(good, dims=img_dim)
    return glm


def file_sane_open(glm_filename):
    """ Open a GLM gridded product file, and make sure it has the
    format we expect. Returns an xarray dataset.
    """
    glm = xr.open_dataset(glm_filename)

    # If the a GLM grid dataset was opened as a time series and saved
    # as a time series NetCDF file, then there will be more than the x
    # and y dimensions of the image. Check for that possiblity.
    fed_dims = list(glm.flash_extent_density.dims)
    if len(fed_dims) > 2:
        print("*** Selecting first time from multi-time GLM file ***")
        fed_dims.remove('x')
        fed_dims.remove('y')
        first_time = {}
        for d in glm.flash_extent_density.dims:
            first_time[d] = 0
        glm = glm[first_time]
    return glm



def process_one_glm_file(glm_filename, stack_order = ('y', 'x'),
        thin_to_group_centroids=False, ltgellipsever=1):
    """
    Open one GLM file and return a boolean 2D mask array where lats and lons
    are within the field of view and, optionally, at locations where group
    centroid density is nonzero.

    """

    glm = file_sane_open(glm_filename)

    # Add parallax-corrected lon, lat to the GLM dataset
    glm = glm_locations(glm, ltgellipsever=ltgellipsever)

    # Get the 2D boolean mask giving the region of good data.
    good = glm['in_fov']

    # Apply any other criteria here - for instance, thin with a random subset
    # of the good array, every 16th or 25th pixel corresponding to one ob
    # per unique GLM pixel of 8 km (nadir, 4x4 2 km pixels) or 10 km
    # (over best part of conus, 5x5 2 km pixels).

    # thin images using group centroids
    if thin_to_group_centroids:
        good = good & (glm.group_centroid_density > 0)

    # Use the 2D boolean mask to turn the 2D images into a flat collection
    # of pixels for each variable.
    glm_px = glm.stack(pixel=stack_order)
    good_px = good.stack(pixel=stack_order)
    return glm, glm_px[{'pixel':good_px}]


if __name__ == '__main__':

    # === configurable parameters ===

    # Version of the lightning ellipse to use
    ltgellipsever = 1

    thin_to_group_centroids=True

    # === end config

    glm_files = sys.argv[1:]
    for glm_file in glm_files:
        glm, glm_px = process_one_glm_file(glm_file,
            thin_to_group_centroids=thin_to_group_centroids,
            ltgellipsever=ltgellipsever)

        # These are 1D arrays of pixel locations, and any GLM variable
        # can be selected here (e.g., minimum_flash_area, or total_energy)
        # To get a numpy array instead of an xarray.DataArray, add ".data"
        fed_px = glm_px.flash_extent_density
        lon_px = glm_px.lon_ltg_ellps
        lat_px = glm_px.lat_ltg_ellps

        # How many obs did we have in each file?
        print(glm_file, glm_px.dims['pixel'])

        # See the process_one_glm_file function for advice on thinning.

        # Write out fhe fed_px, lon_px, lat_px as observations here