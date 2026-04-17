# main.py
import cv2
from camera_input import CameraInput, preprocess_frame


def main():
    camera = CameraInput(flip=False)

    try:
        camera.open()

        while True:
            frame = camera.get_frame()

            if frame is None:
                print("Failed to read frame")
                break

            # Preprocess (for future detector)
            processed = preprocess_frame(frame)

            # Show original
            cv2.imshow("Camera Feed", frame)

            # Show processed (convert back for display)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            cv2.imshow("Processed (for model)", processed_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()