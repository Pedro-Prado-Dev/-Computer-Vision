import dataclasses
import time
from datetime import datetime

import cv2
import torch
import numpy as np

from src.typings.draw import Border, Text
from src.video.exceptions import VideoIsNotRecognizedException, ScreenshotNotTokeException
from src.video.utils.frame import FrameUtils
from src.audio.audio import Speaker
from src.typings.points import Point


@dataclasses.dataclass
class DistanceInterval:
    min_distance: float
    max_distance: float

    def isin(self, distance: float) -> bool:
        return eval(f"{self.max_distance} >= {distance} >= {self.min_distance}")


class Recognition:
    def __init__(self, spoke_immediately: bool = False):
        self._spoke_immediately = spoke_immediately

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.video = cv2.VideoCapture(0)

        self.__available_distances_intervals_to_speak = [
            DistanceInterval(1, 3),
            DistanceInterval(6, 8),
        ]

        self.speaker = Speaker()

        if not self.video.isOpened():
            raise VideoIsNotRecognizedException('Video not opened')

    def _get_video_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        return self.video.read()

    @staticmethod
    def _show_processed_frame(frame: cv2.typing.MatLike):
        cv2.imshow('YOLOv5 Object Detection', frame)

    @staticmethod
    def _can_stop_process():
        return cv2.waitKey(1) & 0xFF == ord('q')

    @staticmethod
    def _draw_boxes(
        frame: np.ndarray,
        text_to_show: str,
        points: list[Point],
        border: Border = Border(color=(0, 255, 0), thickness=2),
        text: Text = Text(color=(36, 255, 12), thickness=2, scale=0.9)
    ) -> None:
        cv2.rectangle(frame, points[0], points[1], border.color, border.thickness)

        text_points = (points[0][0], points[0][1] - 10)
        cv2.putText(
            frame,
            text_to_show,
            text_points,
            cv2.FONT_HERSHEY_SIMPLEX,
            text.scale,
            text.color,
            text.thickness
        )

    def run(self):
        last_time_spoken = time.time() - 2
        old_distance = -1

        while True:
            has_could_take_screenshot, frame = self._get_video_frame()
            if not has_could_take_screenshot:
                raise ScreenshotNotTokeException('Error while getting video frame')

            rgb_image = FrameUtils.bgr2rgb(frame)
            results = self.model(rgb_image)

            boxes = results.xyxy[0].cpu().numpy()
            labels = results.names

            speak = True
            new_distance = 0

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                label = labels[int(cls)]

                distance_of_object = 650 / (x2 - x1 + y2 - y1)
                distance_of_object_text = f"Distance: {distance_of_object:.2f} meters"

                if label == "person":
                    
                    if distance_of_object and abs(old_distance - distance_of_object) > 0.5:
                        current_time = time.time()
                        current_time_spoken = current_time - last_time_spoken
                        if current_time_spoken >= 2:
                            self.speaker.speak(distance_of_object_text)
                            last_time_spoken = current_time_spoken
                            print(f"OLD DISTANCE = {old_distance:.2f} - {label}")
                            old_distance = distance_of_object

                            print(f"NEW DISTANCE = {distance_of_object:.2f} - {label}")

                    else:
                        last_time_spoken = time.time() - 2

                        points_to_draw = list(
                            map(
                                lambda point: (
                                    int(point[0]),
                                    int(point[1]),
                                ),
                                [(x1, y1), (x2, y2)]
                            )
                        )

                        self._draw_boxes(frame, str(f"{old_distance:.2f} - {label}"), points_to_draw)


                else:
                    last_time_spoken = time.time() - 2

            self._show_processed_frame(frame)
            if self._can_stop_process():
                break

        self.video.release()
        cv2.destroyAllWindows()
