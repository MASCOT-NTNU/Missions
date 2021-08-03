#!/usr/bin/python
# GoogleMapDownloader.py 
# Created by Hayden Eskriett [http://eskriett.com]
# Modified by Yaolin Ge
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/

import urllib.request
# import Image
from PIL import Image
import os
import math


class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12):
        """
            GoogleMapDownloader Constructor

            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude 
            and zoom level

            Returns:    An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom
        # print(numTiles)

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
                    tile_size / (2 * math.pi))) * numTiles // tile_size
        print(point_x, point_y)
        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                url = 'https://mt0.google.com/vt?x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                    self._zoom)

                current_tile = str(x) + '-' + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img


def main():
    # Create a new instance of GoogleMap Downloader
    # gmd = GoogleMapDownloader(51.5171, 0.1062, 13)
    gmd = GoogleMapDownloader(63.446905, 10.419426, 13)

    print("The tile coorindates are {}".format(gmd.getXY()))

    try:
        # Get the high resolution image
        img = gmd.generateImage()
    except IOError:
        print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
    else:
        # Save the image to disk
        img.save("high_resolution_image.png")
        print("The map has successfully been created")


if __name__ == '__main__':  main()


# Therefore:
ZOOM0_SIZE = 512  # Not 256

# Geo-coordinate in degrees => Pixel coordinate
def g2p(lat, lon, zoom):
    return (
        # x
        ZOOM0_SIZE * (2 ** zoom) * (1 + lon / 180) / 2,
        # y
        ZOOM0_SIZE / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
    )


# Pixel coordinate => geo-coordinate in degrees
def p2g(x, y, zoom):
    return (
        # lat
        (atan(exp(pi - y / ZOOM0_SIZE * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
        # lon
        (x / ZOOM0_SIZE * 2 / (2 ** zoom) - 1) * 180,
    )

#%%

# RA, 2019-01-16
# Download a rendered map from Mapbox based on a bounding box
# License: CC0 -- no rights reserved

import io
import urllib.request
from PIL import Image
from math import pi, log, tan, exp, atan, log2, floor
import matplotlib
matplotlib.use('agg')
# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
# Note on google API:
# The world map is obtained with lat=lon=0, w=h=256, zoom=0
# Note on mapbox API:
# The world map is obtained with lat=lon=0, w=h=512, zoom=0
#
# Therefore:
ZOOM0_SIZE = 512  # Not 256

# Geo-coordinate in degrees => Pixel coordinate
def g2p(lat, lon, zoom):
    return (
        # x
        ZOOM0_SIZE * (2 ** zoom) * (1 + lon / 180) / 2,
        # y
        ZOOM0_SIZE / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
    )


# Pixel coordinate => geo-coordinate in degrees
def p2g(x, y, zoom):
    return (
        # lat
        (atan(exp(pi - y / ZOOM0_SIZE * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
        # lon
        (x / ZOOM0_SIZE * 2 / (2 ** zoom) - 1) * 180,
    )


# axis to mapbox
def ax2mb(left, right, bottom, top):
    return (left, bottom, right, top)


# mapbox to axis
def mb2ax(left, bottom, right, top):
    return (left, right, bottom, top)


# bbox = (left, bottom, right, top) in degrees
def get_map_by_bbox(bbox):
    # Token from https://www.mapbox.com/api-documentation/maps/#static
    token = "pk.eyJ1IjoiYnVzeWJ1cyIsImEiOiJjanF4cXNoNmEwOG1yNDNycGw5bTByc3g5In0.flzpO633oGAY5aa-RQa4Ow"

    # The region of interest in geo-coordinates in degrees
    # For example, bbox = [120.2206, 22.4827, 120.4308, 22.7578]
    (left, bottom, right, top) = bbox

    # Sanity check
    assert (-90 <= bottom < top <= 90)
    assert (-180 <= left < right <= 180)

    # Rendered image map size in pixels as it should come from MapBox (no retina)
    (w, h) = (1024, 1024)

    # The center point of the region of interest
    (lat, lon) = ((top + bottom) / 2, (left + right) / 2)

    # Reduce precision of (lat, lon) to increase cache hits
    snap_to_dyadic = (lambda a, b: (lambda x, scale=(2 ** floor(log2(abs(b - a) / 4))): (round(x / scale) * scale)))

    lat = snap_to_dyadic(bottom, top)(lat)
    lon = snap_to_dyadic(left, right)(lon)

    assert ((bottom < lat < top) and (left < lon < right)), "Reference point not inside the region of interest"

    # Look for appropriate zoom level to cover the region of interest
    for zoom in range(16, 0, -1):
        # Center point in pixel coordinates at this zoom level
        (x0, y0) = g2p(lat, lon, zoom)

        # The "container" geo-region that the downloaded map would cover
        (TOP, LEFT) = p2g(x0 - w / 2, y0 - h / 2, zoom)
        (BOTTOM, RIGHT) = p2g(x0 + w / 2, y0 + h / 2, zoom)

        # Would the map cover the region of interest?
        if (LEFT <= left < right <= RIGHT):
            if (BOTTOM <= bottom < top <= TOP):
                break

    # Collect all parameters
    params = {
        'style': "streets-v10",
        'lat': lat,
        'lon': lon,
        'token': token,
        'zoom': zoom,
        'w': w,
        'h': h,
        'retina': "@2x",
    }

    url_template = "https://api.mapbox.com/styles/v1/mapbox/{style}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=false&logo=false"
    url = url_template.format(**params)

    # Download the rendered image
    with urllib.request.urlopen(url) as response:
        j = Image.open(io.BytesIO(response.read()))

    # If the "retina" @2x parameter is used, the image is twice the size of the requested dimensions
    (W, H) = j.size
    assert ((W, H) in [(w, h), (2 * w, 2 * h)])

    # Extract the region of interest from the larger covering map
    i = j.crop((
        round(W * (left - LEFT) / (RIGHT - LEFT)),
        round(H * (top - TOP) / (BOTTOM - TOP)),
        round(W * (right - LEFT) / (RIGHT - LEFT)),
        round(H * (bottom - TOP) / (BOTTOM - TOP)),
    ))

    return i


def test():
    bbox = [120.2206, 22.4827, 120.4308, 22.7578]
    map = get_map_by_bbox(bbox)

    import matplotlib as mpl
    mpl.use("Agg")

    import matplotlib.pyplot as plt
    plt.imshow(map, extent=mb2ax(*bbox))
    plt.show()


if __name__ == "__main__":
    test()


#%%
import matplotlib.pyplot as plt
import numpy as np
a = plt.imread("map.png")
d = np.array(a[:720, 352:])
YLIM = [63.4432, 63.4603]
XLIM = [10.3822, 10.4388]
# b = a[:720, 349:, 0] + a[:720, 349:, 1] + a[:720, 349:, 2]
    # + a[:720, 349:, 3]
# plt.imshow(a[:720, 349:], extent = [XLIM[0], XLIM[1], YLIM[0], YLIM[1]], aspect = 3)
# plt.imshow(b / np.amax(b), extent = [XLIM[0], XLIM[1], YLIM[0], YLIM[1]], aspect = 3)
# plt.imshow(d)
plt.imshow(d[:, :, 0])
plt.show()


##%%
# LATLIM = [63.4432, 63.4603]
# LONLIM = [10.3822, 10.4388]
# img = np.array(img)
# # img = np.array(img[:720, 352:])
# lat_map = np.linspace(LATLIM[0], LATLIM[1], img.shape[0])
# lon_map = np.linspace(LONLIM[0], LONLIM[1], img.shape[1])
# # lat_mapv = lat_mapm.reshape(-1, 1)
# # lon_mapv = lon_mapm.reshape(-1, 1)
# col = []
# lat_mapv = []
# lon_mapv = []
# downsample_factor = 10
# for i in range(0, img.shape[0], downsample_factor):
#     for j in range(0, img.shape[1], downsample_factor):
#         lat_mapv.append(lat_map[i])
#         lon_mapv.append(lon_map[j])
#         col.append(img[i, j, 0])
#        # col.append((img[i, j, 0] + img[i, j, 1] + img[i, j, 2]) / 3)
fig.add_trace(
    go.Isosurface(x=lon_mapv, y=lat_mapv, z=np.zeros_like(lat_mapv).squeeze(),
                  # value=mu_prior[ind], coloraxis="coloraxis"),
                  value=col, coloraxis="coloraxis"),
    # value=mu_cond[ind], coloraxis="coloraxis"),
    row=1, col=1
)
