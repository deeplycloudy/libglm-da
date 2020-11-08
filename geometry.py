import numpy as np
import pyproj as proj4
from lmatools.grid.fixed import get_GOESR_coordsys
from lmatools.coordinateSystems import MapProjection, GeographicSystem, CoordinateSystem
from glmtools.io.lightning_ellipse import lightning_ellipse_rev

def semiaxes_to_invflattening(semimajor, semiminor):
    """ Calculate the inverse flattening from the semi-major
        and semi-minor axes of an ellipse"""
    rf = semimajor/(semimajor-semiminor)
    return rf

class GeostationaryFixedGridSystemAltEllipse(CoordinateSystem):

    def __init__(self, subsat_lon=0.0, subsat_lat=0.0, sweep_axis='y',
                 sat_ecef_height=35785831.0,
                 semimajor_axis=None,
                 semiminor_axis=None,
                 datum='WGS84'):
        """
        Satellite height is with respect to an arbitray ellipsoid whose
        shape is given by semimajor_axis (equatorial) and semiminor_axis(polar)

        Fixed grid coordinates are in radians.
        """
        rf = semiaxes_to_invflattening(semimajor_axis, semiminor_axis)
        # print("Defining alt ellipse for Geostationary with rf=", rf)
        self.ECEFxyz = proj4.Proj(proj='geocent',
            a=semimajor_axis, rf=rf)
        self.fixedgrid = proj4.Proj(proj='geos', lon_0=subsat_lon,
            lat_0=subsat_lat, h=sat_ecef_height, x_0=0.0, y_0=0.0,
            units='m', sweep=sweep_axis,
            a=semimajor_axis, rf=rf)
        self.h=sat_ecef_height

    def toECEF(self, x, y, z):
        X, Y, Z = x*self.h, y*self.h, z*self.h
        return proj4.transform(self.fixedgrid, self.ECEFxyz, X, Y, Z)

    def fromECEF(self, x, y, z):
        X, Y, Z = proj4.transform(self.ECEFxyz, self.fixedgrid, x, y, z)
        return X/self.h, Y/self.h, Z/self.h

class GeographicSystemAltEllps(CoordinateSystem):
    """
    Coordinate system defined on the surface of the earth using latitude,
    longitude, and altitude, referenced by default to the WGS84 ellipse.

    Alternately, specify the ellipse shape using an ellipse known
    to pyproj, or [NOT IMPLEMENTED] specify r_equator and r_pole directly.
    """
    def __init__(self, ellipse='WGS84', datum='WGS84',
                 r_equator=None, r_pole=None):
        if (r_equator is not None) | (r_pole is not None):
            rf = semiaxes_to_invflattening(r_equator, r_pole)
            # print("Defining alt ellipse for Geographic with rf", rf)
            self.ERSlla = proj4.Proj(proj='latlong', #datum=datum,
                                     a=r_equator, rf=rf)
            self.ERSxyz = proj4.Proj(proj='geocent', #datum=datum,
                                     a=r_equator, rf=rf)
        else:
            # lat lon alt in some earth reference system
            self.ERSlla = proj4.Proj(proj='latlong', ellps=ellipse, datum=datum)
            self.ERSxyz = proj4.Proj(proj='geocent', ellps=ellipse, datum=datum)
    def toECEF(self, lon, lat, alt):
        projectedData = np.array(proj4.transform(self.ERSlla, self.ERSxyz, lon, lat, alt ))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]

    def fromECEF(self, x, y, z):
        projectedData = np.array(proj4.transform(self.ERSxyz, self.ERSlla, x, y, z ))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]


def get_GOESR_coordsys_alt_ellps(sat_lon_nadir=-75.0, ellipse_ver=1):
    goes_sweep = 'x' # Meteosat is 'y'
    datum = 'WGS84'
    sat_ecef_height=35786023.0

    # equatorial and polar radii
    ltg_ellps_re, ltg_ellps_rp = lightning_ellipse_rev[ellipse_ver]

    geofixcs = GeostationaryFixedGridSystemAltEllipse(subsat_lon=sat_lon_nadir,
                    semimajor_axis=ltg_ellps_re, semiminor_axis=ltg_ellps_rp,
                    datum=datum, sweep_axis=goes_sweep,
                    sat_ecef_height=sat_ecef_height)
    grs80lla = GeographicSystemAltEllps(r_equator=ltg_ellps_re, r_pole=ltg_ellps_rp,
                                datum='WGS84')
    return geofixcs, grs80lla

def get_glm_earth_geometry(glm, nadir, ltgellipsever):
    x_1d = glm.x
    y_1d = glm.y
    x,y = np.meshgrid(x_1d, y_1d) # Two 2D arrays of fixed grid coordinates
    z=np.zeros_like(x)

    # Figure out the approximate GLM earth coverage based on GOES data book.
    # Convert below to radians and plot in fixed grid coordinates.
    #     rect_span = np.radians(14.81)*satheight #minimum degrees
    #     circ_diam = np.radians(15.59)*satheight #minimum degrees
    rect_span = np.radians(15) #minimum degrees - matches figure in GOES data book
    circ_diam = np.radians(16) #minimum degrees
    glm_angle = np.sqrt(x*x + y*y)
    outside_glm_full_disk = ((np.abs(x) > rect_span/2.0) |
                             (np.abs(y) > rect_span/2.0) |
                             (glm_angle > circ_diam/2.0) )


    geofixCS, grs80lla = get_GOESR_coordsys(nadir)
    geofix_ltg, lla_ltg = get_GOESR_coordsys_alt_ellps(nadir, ltgellipsever)
    X,Y,Z = geofix_ltg.toECEF(x,y,z)

    # X, Y, Z = geofix_ltg.toECEF(x,y,z) gives the 3D, earth-centered,
    # earth-fixed position of the intersection of the satellite-relative fixed
    # grid angle with the lightning ellipse.

    # The 3D position (X,Y,Z) defines an implicit lon, lat, alt with respect to
    # the target earth surface. Here, we intersect with the grs80 earth surface.

    lon_ltg,lat_ltg,alt_ltg=grs80lla.fromECEF(X,Y,Z)
    lon_ltg.shape = x.shape
    lat_ltg.shape = y.shape

    return (x, y), (X, Y, Z), (lon_ltg, lat_ltg, alt_ltg), outside_glm_full_disk


def get_geometry_hrrr(glm, nadir, hrrr, ltgellipsever=1):

    # We don't use X, Y, Z below.
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
     outside_glm_full_disk) = get_glm_earth_geometry(glm, nadir, ltgellipsever)

    ### HRRR interpolation ###

    # Everything below presumes LCC.
    assert hrrr.MAP_PROJ_CHAR == 'Lambert Conformal'

    corner_0_lla = (hrrr.XLONG[0,0,0].data,
                    hrrr.XLAT[0,0,0].data,
                    np.asarray(0.0, dtype=hrrr.XLAT[0,0,0].dtype))
    corner_1_lla = (hrrr.XLONG[0,-1,-1].data,
                    hrrr.XLAT[0,-1,-1].data,
                    np.asarray(0.0, dtype=hrrr.XLAT[0,1,-1].dtype))

    hrrr_dx, hrrr_dy = hrrr.DX, hrrr.DX
    hrrr_Nx, hrrr_Ny = hrrr.dims['west_east'], hrrr.dims['south_north']

    hrrrproj={
    'lat_0':hrrr.CEN_LAT,
    'lon_0':hrrr.CEN_LON+360.0,
    'lat_1':hrrr.TRUELAT1,
    'lat_2':hrrr.TRUELAT2,
    # 'R':hrrr.LambertConformal_Projection.earth_radius,
    # 'a':6371229,
    # 'b':6371229,
    'R':6371229,
    }

    lcc = MapProjection(projection='lcc',
                        ctrLat=hrrrproj['lat_0'],
                        ctrLon=hrrrproj['lon_0'], **hrrrproj)
    hrrr_lla = GeographicSystem(r_equator=hrrrproj['R'], r_pole=hrrrproj['R'])

    lcc_cornerx_0, lcc_cornery_0, lcc_cornerz_0 = lcc.fromECEF(
                                                    *hrrr_lla.toECEF(*corner_0_lla))
    lcc_cornerx_1, lcc_cornery_1, lcc_cornerz_1 = lcc.fromECEF(
                                                    *hrrr_lla.toECEF(*corner_1_lla))



# def grid_idx(x, y, x0, y0, dx, dy):
#     """
#     Convert x, y returned by [projection].fromECEF to the grid index in the
#     NetCDF file. x0 and y0 are the [projection] coordinates of the center of
#     the zero-index position in the NetCDF grid. dx and dy are the grid spacing
#     in meters.
#
#     returns (xidx, yidx)
#     Taking int(xidx) will give the zero-based grid cell index.
#     """
#     # To get the correct behavior with int(xidx), add a half
#     # since x0 is the center.
#     xidx = (x-x0)/dx + 0.5
#     yidx = (y-y0)/dy + 0.5
#     return xidx, yidx



# The 3D position (X,Y,Z) defines an implicit lon, lat, alt with respect to the
# spherical earth we specified for the HRRR and its associated Lambert
# conformal projection. We let proj4 handle the mapping from the ECEF
# coordinates (an absolute position) directly to LCC.

    lccx2, lccy2, lccz2 = lcc.fromECEF(*hrrr_lla.toECEF(lon_ltg,
                                        lat_ltg, np.zeros_like(lon_ltg)))
    lccx2.shape=x.shape
    lccy2.shape=x.shape
    lccx=lccx2
    lccy=lccy2

    # Set up the model grid, since the hrrr file doesn't include those values.

    hrrrx_1d = np.arange(hrrr_Nx, dtype='f4') * hrrr_dx + lcc_cornerx_0
    hrrry_1d = np.arange(hrrr_Ny, dtype='f4') * hrrr_dy + lcc_cornery_0
    hrrrx, hrrry = np.meshgrid(hrrrx_1d, hrrry_1d)
    interp_loc = np.vstack((hrrrx.flatten(), hrrry.flatten())).T


# GLM variables are filled with nan everywhere there is no lightning,
# so set those locations corresponding to valid earth locations to zero.

    lcc_glm_x_flat = lccx[:,:].flatten()
    lcc_glm_y_flat = lccy[:,:].flatten()

    good = np.isfinite(lcc_glm_x_flat) & np.isfinite(lcc_glm_y_flat)
    good = good & (~outside_glm_full_disk.flatten())
    data_loc = np.vstack((lcc_glm_x_flat[good], lcc_glm_y_flat[good])).T

    return(data_loc, good, interp_loc, hrrrx.shape)

