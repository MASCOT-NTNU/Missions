import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import os
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})


class MaretecDataHandler:
    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-14_2021-09-15/WaterProperties.hdf5"
    delft_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Polygon/fig/"
    polygon_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    circumference = 40075000 # [m], circumference of the earth

    def __init__(self):
        self.lat_origin, self.lon_origin = 41.061874, -8.650977  # origin location
        pass

    def loaddata(self):
        print("Now it will load the Maretec data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.grid = self.data.get('Grid')
        self.lat = np.array(self.grid.get("Latitude"))[:-1, :-1]
        self.lon = np.array(self.grid.get("Longitude"))[:-1, :-1]
        self.depth = []
        self.salinity = []
        for i in range(1, 26):
            string_z = "Vertical_{:05d}".format(i)
            string_sal = "salinity_{:05d}".format(i)
            self.depth.append(np.mean(np.array(self.grid.get("VerticalZ").get(string_z)), axis = 0))
            self.salinity.append(np.mean(np.array(self.data.get("Results").get("salinity").get(string_sal)), axis = 0))
        self.depth = np.array(self.depth)
        self.salinity = np.array(self.salinity)
        t2 = time.time()
        print("Data is loaded correctly, time consumed: ", t2 - t1)

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def visualiseData(self):
        self.checkFolder()
        print("Here it comes the plotting for the updated results from Maretec.")
        files = os.listdir(self.data_path[:81])
        files.sort()
        counter = 0
        for i in range(len(files)):
            if files[i] != ".DS_Store":
                datapath = self.data_path[:81] + files[i] + "/WaterProperties.hdf5"
                self.data_path = datapath
                self.loaddata()
                for j in range(self.salinity.shape[0]):
                    print(j)
                    plt.figure(figsize=(10, 10))
                    plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
                                c=self.salinity[j, :self.lon.shape[1], :], vmin=26, vmax=36, cmap="Paired")
                    plt.colorbar()
                    plt.title("Surface salinity estimation from Maretec at time: {:02d}:00 during ".format(
                        j) + self.data_path[81:102])
                    plt.xlabel("Lon [deg]")
                    plt.ylabel("Lat [deg]")
                    plt.savefig(self.figpath + "/P_{:04d}.png".format(counter))
                    plt.close("all")
                    counter = counter + 1
                # plt.show()

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = MaretecDataHandler.deg2rad((lat - lat_origin)) / 2 / np.pi * MaretecDataHandler.circumference
        y = MaretecDataHandler.deg2rad((lon - lon_origin)) / 2 / np.pi * MaretecDataHandler.circumference * np.cos(MaretecDataHandler.deg2rad(lat))
        return x, y

    def getPolygonArea(self):
        area = 0
        prev = self.polygon[-1]
        for i in range(self.polygon.shape[0] - 1):
            now = self.polygon[i]
            xnow, ynow = MaretecDataHandler.latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = MaretecDataHandler.latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2 / 1e6
        return self.PolyArea
        # print("Area: ", self.PolyArea / 1e6, " km2")

    def loadDelft3D(self, wind_dir, wind_level):
        delft3dpath = self.delft_path + wind_dir + "_" + wind_level + "_all.h5"
        delft3d = h5py.File(delft3dpath, 'r')
        self.lat_delft3d = np.array(delft3d.get("lat"))[:, :, 0]
        self.lon_delft3d = np.array(delft3d.get("lon"))[:, :, 0]
        self.salinity_delft3d = np.array(delft3d.get("salinity"))[:, :, 0]

    def plotdataonDay(self, day, hour, wind_dir, wind_level):
        print("This will plot the data on day " + day)
        datapath = self.data_path[:81] + day + "/WaterProperties.hdf5"
        hour_start = int(hour[:2])
        hour_end = int(hour[3:])
        self.data_path = datapath
        self.loaddata()
        self.loadDelft3D(wind_dir, wind_level)
        plt.figure(figsize=(10, 10))
        plt.scatter(self.lon_delft3d, self.lat_delft3d, c=self.salinity_delft3d, vmin=26, vmax=36, alpha=1,
                    cmap="Paired")
        plt.colorbar()
        plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
                    c=self.salinity[hour_start, :self.lon.shape[1], :], vmin=26, vmax=36, alpha = .25, cmap="Paired")
        plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
                    c=self.salinity[hour_end, :self.lon.shape[1], :], vmin=26, vmax=36, alpha=.05, cmap="Paired")
        plt.title("Surface salinity estimation from Maretec during " + self.data_path[81:102])
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        polygon = plt.ginput(n = 100) # wait for the click to select the polygon
        plt.show()
        self.polygon = []
        for i in range(len(polygon)):
            self.polygon.append([polygon[i][1], polygon[i][0]])
        self.polygon = np.array(self.polygon)
        np.savetxt(self.polygon_path + "polygon.txt", self.polygon, delimiter=", ")
        print("Ploygon is selected successfully. Total area: ", self.getPolygonArea())
        os.system("say Congrats, Polygon is selected successfully, total area is {:.2f} km2".format(self.getPolygonArea()))
        print("Total area: ", self.getPolygonArea())
        self.save_wind_condition(wind_dir, wind_level)

    def save_wind_condition(self, wind_dir, wind_level):
        f_wind = open(self.polygon_path + "wind_condition.txt", 'w')
        f_wind.write("wind_dir=" + wind_dir + ", wind_level=" + wind_level)
        f_wind.close()
        print("wind_condition is saved successfully!")

if __name__ == "__main__":
    a = MaretecDataHandler()
    pycharm = False
    if pycharm == True:
        a.visualiseData()
    else:
        from Grid import GridPoly
        string_date = "2021-09-14_2021-09-15"
        string_hour = "08_12"
        wind_dir = "North" # [North, East, West, South]
        wind_level = "Moderate" # [Mild, Moderate, Heavy]
        a.plotdataonDay(string_date, string_hour, wind_dir, wind_level)
        b = GridPoly(polygon = a.polygon, debug = False)
        os.system("open /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/MapPlot/map.html")


