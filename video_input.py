import cv2


class VideoInput:
    def __init__(self, video_path, width=640, height=480, loop=False):

        self.video_path = video_path
        self.width = width
        self.height = height
        self.loop = loop
        self.cap = None

    def open(self):


        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open video: {self.video_path}")

    def get_frame(self):

        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        ret, frame = self.cap.read()

        # If video ended
        if not ret:
            if self.loop:
                # Restart video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.get_frame()
            else:
                return None

        # Resize for consistency
        frame = cv2.resize(frame, (self.width, self.height))

        return frame  
    

    def release(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None