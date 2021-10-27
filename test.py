import numpy as np
from mayavi.mlab import *

def test_points3d():
    x = np.arange(5)
    y = np.arange(5)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = np.zeros_like(xx)

    return points3d(xx, yy, zz, np.random.randn(len(xx)), colormap="copper", scale_factor=.25)

# import numpy
# from mayavi.mlab import *

def test_imshow():
    """ Use imshow to visualize a 2D 10x10 random array.
    """
    s = np.random.random((10, 10))
    return imshow(s, colormap='gist_earth')
# test_points3d()


def test_contour3d():
    x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]

    scalars = x * x * 0.5 + y * y + z * z * 2.0

    obj = contour3d(scalars, contours=4, transparent=True)
    return obj


# test_imshow()
test_contour3d(

)