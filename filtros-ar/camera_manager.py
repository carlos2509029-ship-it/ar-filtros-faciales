import cv2


class CameraManager:

    @staticmethod
    def list_available(max_cameras=5):

        cameras = []

        for i in range(max_cameras):

            cap = cv2.VideoCapture(i)

            if cap.isOpened():

                ret, _ = cap.read()

                if ret:
                    cameras.append(i)

            cap.release()

        return cameras

    def __init__(self, camera_index=0):

        self.cap = cv2.VideoCapture(camera_index)

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return cv2.flip(frame, 1)

    def release(self):

        self.cap.release()
        cv2.destroyAllWindows()