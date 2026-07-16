class Robot:

    def get_direction(self, center_x, frame_width):

        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        if center_x < left_boundary:
            return "TURN LEFT"

        elif center_x > right_boundary:
            return "TURN RIGHT"

        return "MOVE FORWARD"


    def estimate_distance(self, area):

        if area < 3000:
            return "FAR"

        elif area < 10000:
            return "MEDIUM"

        return "CLOSE"