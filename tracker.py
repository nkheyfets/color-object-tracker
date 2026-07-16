import cv2
import time
from collections import deque
from datetime import datetime
from robot import Robot
from config import (
    LOWER_BLUE,
    UPPER_BLUE,
    MINIMUM_AREA,
)


def main() -> None:
    camera = cv2.VideoCapture(0)
    robot = Robot()
    position_history = deque(maxlen=50)
    previous_time = time.time()

    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Webcam started. Press Q to close the window.")

    while True:
        success, frame = camera.read()


        if not success:
            print("Error: Could not read a frame from the webcam.")
            break

        current_time = time.time()
        elapsed_time = current_time - previous_time

        if elapsed_time > 0:
            fps = 1 / elapsed_time
        else:
            fps = 0

        previous_time = current_time

        frame_height, frame_width = frame.shape[:2]

        direction = "STOP - OBJECT NOT FOUND"
        distance_status = "UNKNOWN"

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = (100, 100, 100)
        upper_blue = (130, 255, 255)

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

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > MINIMUM_AREA:
                x, y, width, height = cv2.boundingRect(largest_contour)
                distance_status = robot.estimate_distance(area)
                center_x = x + width // 2
                center_y = y + height // 2

                position_history.append((center_x, center_y))

                direction = robot.get_direction(center_x, frame_width)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 0, 255),
                    -1
                )


        for index in range(1, len(position_history)):
            previous_point = position_history[index - 1]
            current_point = position_history[index]

            cv2.line(
                frame,
                previous_point,
                current_point,
                (0, 255, 255),
                2
            )
        
        cv2.putText(
            frame,
            direction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {distance_status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (frame_width // 3, 0),
            (frame_width // 3, frame_height),
            (255, 255, 255),
            1
        )

        cv2.line(
            frame,
            (2 * frame_width // 3, 0),
            (2 * frame_width // 3, frame_height),
            (255, 255, 255),
            1
)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Color Object Tracker", frame)
        cv2.imshow("HSV Image", hsv)
        cv2.imshow("Blue Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            position_history.clear()
            print("Trail cleared")

        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tracker_screenshot_{timestamp}.png"

            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")



if __name__ == "__main__":
    main()