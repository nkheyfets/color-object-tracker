import cv2

from config import LOWER_BLUE, UPPER_BLUE, MINIMUM_AREA


class Detector:

    def detect(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            LOWER_BLUE,
            UPPER_BLUE
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest)

        if area < MINIMUM_AREA:
            return None, mask

        x, y, width, height = cv2.boundingRect(largest)

        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area": area,
            "center_x": x + width // 2,
            "center_y": y + height // 2,
        }, mask