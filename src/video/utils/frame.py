import cv2


class FrameUtils:
    @staticmethod
    def bgr2rgb(image: cv2.typing.MatLike):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
