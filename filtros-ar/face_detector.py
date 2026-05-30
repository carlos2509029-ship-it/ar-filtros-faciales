import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(
        self,
        static_mode=False,
        max_faces=1,
        refine_landmarks=True,
        detection_confidence=0.5,
        tracking_confidence=0.5
    ):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.drawer = mp.solutions.drawing_utils

        self.styles = mp.solutions.drawing_styles

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        return results

    def draw(self, frame, results):

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                self.drawer.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.styles.get_default_face_mesh_tesselation_style()
                )

        return frame

    def landmark_xy(self, frame, landmark):

        h, w, _ = frame.shape

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        return (x, y)
    