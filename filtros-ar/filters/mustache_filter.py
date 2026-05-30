import cv2

from .base_filter import BaseFilter


class MustacheFilter(BaseFilter):

    def __init__(
        self,
        png_path,
        target_width_px=180,
        y_offset_px=15
    ):

        self.overlay_rgba = cv2.imread(
            png_path,
            cv2.IMREAD_UNCHANGED
        )

        if self.overlay_rgba is None:
            raise ValueError(
                f"No se pudo cargar la imagen: {png_path}"
            )

        if self.overlay_rgba.shape[2] != 4:
            raise ValueError(
                "La imagen debe ser RGBA (4 canales)"
            )

        self.target_width_px = int(target_width_px)

        self.y_offset_px = int(y_offset_px)

    def apply(self, frame, landmarks):

        nose = landmarks[1]

        h, w, _ = frame.shape

        x_nose = int(nose.x * w)
        y_nose = int(nose.y * h)

        oh, ow = self.overlay_rgba.shape[:2]

        scale = self.target_width_px / ow

        new_w = int(ow * scale)
        new_h = int(oh * scale)

        mustache = cv2.resize(
            self.overlay_rgba,
            (new_w, new_h)
        )

        x = x_nose - new_w // 2

        y = y_nose + self.y_offset_px - new_h // 2

        self.overlay_rgba_on_bgr(
            frame,
            mustache,
            x,
            y
        )

        return frame

    def overlay_rgba_on_bgr(
        self,
        frame,
        rgba,
        x,
        y
    ):

        h, w = frame.shape[:2]

        oh, ow = rgba.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(w, x + ow)
        y2 = min(h, y + oh)

        if x1 >= x2 or y1 >= y2:
            return

        roi = frame[y1:y2, x1:x2]

        crop = rgba[
            y1 - y:y2 - y,
            x1 - x:x2 - x
        ]

        rgb = crop[..., :3]

        alpha = crop[..., 3:4] / 255.0

        blended = alpha * rgb + (1 - alpha) * roi

        roi[:] = blended.astype("uint8")