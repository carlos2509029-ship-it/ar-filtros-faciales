import cv2

from .base_filter import BaseFilter


class GlassesFilter(BaseFilter):

    def __init__(
        self,
        asset_paths,
        scale_factor=2.0,
        y_offset_ratio=0.0
    ):
        self.assets = []

        for path in asset_paths:
            rgba = cv2.imread(
                path,
                cv2.IMREAD_UNCHANGED
            )

            if rgba is not None:
                self.assets.append(rgba)

        self.asset_index = 0

        self.scale_factor = scale_factor
        self.y_offset_ratio = y_offset_ratio

    def next_asset(self):

        if len(self.assets) == 0:
            return

        self.asset_index = (
            self.asset_index + 1
        ) % len(self.assets)

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
        if len(self.assets) == 0:
            return frame

        span, center = self._eye_span_and_center(
            frame,
            landmarks
        )

        rgba = self.assets[
            self.asset_index
        ]

        width = int(
            span * self.scale_factor
        )

        h0, w0 = rgba.shape[:2]

        height = int(
            h0 * width / w0
        )

        rgba = cv2.resize(
            rgba,
            (width, height)
        )

        x = int(
            center[0] - width / 2
        )

        y = int(
            center[1]
            - height / 2
            + height * self.y_offset_ratio
        )

        h, w = frame.shape[:2]

        if (
            x < 0
            or y < 0
            or x + width > w
            or y + height > h
        ):
            return frame

        overlay_rgb = rgba[:, :, :3]

        alpha = rgba[:, :, 3] / 255.0

        roi = frame[
            y:y + height,
            x:x + width
        ]

        for c in range(3):
            roi[:, :, c] = (
                alpha * overlay_rgb[:, :, c]
                + (1 - alpha) * roi[:, :, c]
            )

        frame[
            y:y + height,
            x:x + width
        ] = roi

        return frame