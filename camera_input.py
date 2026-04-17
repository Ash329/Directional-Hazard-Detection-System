import cv2

# input_stream.py
import cv2


class CameraInput:
    def __init__(self, camera_index=0, width=640, height=480, flip=False):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.flip = flip
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() first.")

        ret, frame = self.cap.read()

        if not ret:
            return None

        if self.flip:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def preprocess_frame(frame, target_size=(640, 640)):
    
    ## Resize and RGP for Neural Network 
    
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb