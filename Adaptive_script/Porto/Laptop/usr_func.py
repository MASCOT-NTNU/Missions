import numpy as np
circumference = 40075000 # [m], circumference

def deg2rad(deg):
    return deg / 180 * np.pi

def rad2deg(rad):
    return rad / np.pi * 180

def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = deg2rad((lat - lat_origin)) / 2 / np.pi * circumference
    y = deg2rad((lon - lon_origin)) / 2 / np.pi * circumference * np.cos(deg2rad(lat))
    return x, y

def xy2latlon(x, y, lat_origin, lon_origin):
    lat = lat_origin + rad2deg(x * np.pi * 2.0 / circumference)
    lon = lon_origin + rad2deg(y * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
    return lat, lon
