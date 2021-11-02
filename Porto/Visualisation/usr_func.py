import numpy as np

def angle2anle(angle):
    return 270 - angle

def s2uv(s, angle):
    angle = angle2anle(angle)
    u = s * np.cos(deg2rad(angle))
    v = s * np.sin(deg2rad(angle))
    return u, v

def rad2deg(rad):
    return rad / np.pi * 180

def deg2rad(deg):
    return deg / 180 * np.pi
