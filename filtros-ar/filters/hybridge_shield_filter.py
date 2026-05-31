import cv2
import trimesh
import numpy as np

from .base_filter import BaseFilter


class HybridgeShieldFilter(BaseFilter):

    def __init__(
        self,
        model_path,
        scale_factor=1.2,
        y_offset_ratio=-1.1
    ):

        scene = trimesh.load(
            model_path
        )

        self.mesh = list(
            scene.geometry.values()
        )[0]

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces

        self.scale_factor = scale_factor
        self.y_offset_ratio = y_offset_ratio

    def _eye_span_and_center(
        self,
        frame,
        landmarks
    ):
        h, w = frame.shape[:2]

        left = landmarks[33]
        right = landmarks[263]

        xL = int(left.x * w)
        yL = int(left.y * h)

        xR = int(right.x * w)
        yR = int(right.y * h)

        span = (
            (xR - xL) ** 2 +
            (yR - yL) ** 2
        ) ** 0.5

        center = (
            (xL + xR) // 2,
            (yL + yR) // 2
        )

        return span, center

    def apply(
        self,
        frame,
        landmarks
    ):

        span, center = self._eye_span_and_center(
            frame,
            landmarks
        )

        scale = span * self.scale_factor

        offset_y = int(
            span * self.y_offset_ratio
        )

        pts2d = []

        for v in self.vertices:

            x = int(
                center[0]
                + v[0] * scale
            )

            y = int(
                center[1]
                + offset_y
                - v[1] * scale
            )

            pts2d.append(
                [x, y]
            )

        pts2d = np.array(
            pts2d,
            dtype=np.int32
        )

        for face in self.faces:

            poly = pts2d[face]

            cv2.polylines(
                frame,
                [poly],
                True,
                (0, 255, 255),
                2
            )

        return frame