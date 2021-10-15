
    def load_grid(self):
        print("Loading grid...")
        self.grid_poly = np.loadtxt(self.path_onboard + "grid.txt", delimiter=", ")
        print("grid is loaded successfully, grid shape: ", self.grid_poly.shape)

    def load_polygon(self):
        print("Loading the polygon...")
        self.polygon = np.loadtxt(self.path_onboard + "polygon.txt", delimiter=", ")
        print("Finished polygon loading, polygon: ", self.polygon.shape)

    def plot_3d_prior(self):
        import plotly.graph_objects as go
        import plotly
        prior = self.path_onboard + "Prior_polygon.txt"
        data_prior = np.loadtxt(prior, delimiter=", ")
        depth_prior = data_prior[:, 2]
        lat_prior = data_prior[:, 0]
        lon_prior = data_prior[:, 1]
        salinity_prior = data_prior[:, -1]
        fig = go.Figure(data=[go.Scatter3d(
            x=lon_prior.squeeze(),
            y=lat_prior.squeeze(),
            z=depth_prior.squeeze(),
            mode='markers',
            marker=dict(
                size=12,
                color=salinity_prior.squeeze(),  # set color to an array/list of desired values
                showscale=True,
                coloraxis="coloraxis"
            )
        )])
        fig.update_coloraxes(colorscale="jet")
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/"
        plotly.offline.plot(fig, filename=figpath + "Prior.html", auto_open=True)
