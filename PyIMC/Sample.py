"""
This script shows the sample process of using ROS and IMC to conduct adaptive sampling.
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-28
"""


def get_distance_between_locations(location1, location2):
    return np.sqrt((location1.x - location2.x) ** 2 +
                   (location1.y - location2.y) ** 2 +
                   (location1.z - location2.z) ** 2)  # Note: In practice, x, y need to be found from lat, lon


class Location:
    def __init__(self, lat=None, lon=None, depth=None):
        self.lat = lat
        self.lon = lon
        self.depth = depth


class AUV:
    def __init__(self):
        pass

    def set_waypoint(self, location=None, speed=None):
        self.auv_location = location
        self.auv_speed = speed

    def get_salinity_data(self):
        return self.salinity

    def get_temperature_data(self):
        return self.temperature

    def get_auv_location(self):
        return self.auv_location

    def send_sms(self, message=None):  # Note: can be more generalised, not necessarily only SMS, can be multiple source
        self.SMS(message)

    def is_arrived(self, destination=None):
        if get_distance_between_locations(self.auv_location, destination) < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def pop_up(self):
        self.send_sms()
        pass  # TODO: add pop up functions


class AdaptiveSampler:

    def __init__(self, starting_location=None, speed=None):
        self.starting_location = starting_location
        self.speed = speed
        self.configure_auv()
        self.run()

    def configure_auv(self):
        self.auv = AUV()
        self.auv.set_waypoint(location=self.starting_location, speed=self.speed)
        pass

    def run(self):
        self.next_location = self.starting_location
        for i in range(self.steps):
            if self.auv.is_arrived(self.next_location):
                self.auv.get_salinity_data()
                ''' Path planner block, you don't have to worry about this
                self.update_field(auv_data)
                self.next_location = PathPlanner().get_next_location()
                '''
                self.auv.set_waypoint(self.next_location, self.speed)
                if i % 5 == 0:
                    self.auv.pop_up()


if __name__ == "__main__":
    adaptive_missioner = AdaptiveSampler()

