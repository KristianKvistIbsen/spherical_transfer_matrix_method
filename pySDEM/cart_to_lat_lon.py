import numpy as np
def cart_to_lat_lon(x, y, z):
    """Convert Cartesian coordinates to latitude/longitude for pysh."""
    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle from z-axis
    phi = np.arctan2(y, x)    # azimuthal angle from x-axis
    
    # Convert to lat/lon in degrees
    lat = 90 - np.degrees(theta)  # theta=0 is north pole (lat=90)
    lon = np.degrees(phi)         # phi=0 is along positive x-axis
    
    # Ensure longitude is in [0, 360) range
    lon = np.where(lon < 0, lon + 360, lon)
    
    return r, lat, lon