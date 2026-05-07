import time


class OvertakeManager:

    def __init__(self, world, vehicle):

        self.world = world
        self.vehicle = vehicle

        self.state = "IDLE"

        self.overtake_start_time = 0

    def update(self, lane_detections, vehicle_speed):

        front_vehicle = self._find_front_vehicle(lane_detections)

        # ==================================================
        # START OVERTAKE
        # ==================================================

        if self.state == "IDLE":

            if front_vehicle:

                distance = front_vehicle.get("distance", 999)

                if distance < 15 and vehicle_speed > 10:

                    self.state = "OVERTAKING"

                    self.overtake_start_time = time.time()

        # ==================================================
        # RETURN TO LANE
        # ==================================================

        elif self.state == "OVERTAKING":

            elapsed = time.time() - self.overtake_start_time

            if elapsed > 4.0:

                self.state = "RETURNING"

                self.return_start = time.time()

        # ==================================================
        # COMPLETE
        # ==================================================

        elif self.state == "RETURNING":

            elapsed = time.time() - self.return_start

            if elapsed > 3.0:

                self.state = "IDLE"

        return self.state

    def _find_front_vehicle(self, detections):

        closest = None

        min_distance = 999

        for det in detections:

            obj_class = det.get("class", "")

            if obj_class not in [
                "car",
                "truck",
                "bus",
                "motorcycle"
            ]:
                continue

            distance = det.get("distance", 999)

            if distance < min_distance:

                min_distance = distance

                closest = det

        return closest