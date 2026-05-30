import cv2

from camera_manager import CameraManager
from face_detector import FaceDetector
from filters.mustache_filter import MustacheFilter
from filters.glasses_filter import GlassesFilter


cameras = CameraManager.list_available()

if not cameras:
    print("No se detectaron cámaras disponibles.")
    exit()

print("Cámaras detectadas:")

for cam in cameras:
    print(f"[{cam}] Cámara {cam}")

choice = input(
    "Selecciona cámara (Enter = primera): "
).strip()

selected_camera = (
    cameras[0]
    if choice == ""
    else int(choice)
)

if selected_camera not in cameras:
    print("La cámara seleccionada no está disponible.")
    exit()


camera = CameraManager(selected_camera)

detector = FaceDetector()

mustache_filter = MustacheFilter(
    "filtros-ar/assets/mustache.png"
)

glasses_filter = GlassesFilter(
    [
        "filtros-ar/assets/glasses.png",
        "filtros-ar/assets/glasses2.png"
    ]
)


while True:

    frame = camera.get_frame()

    if frame is None:
        break

    results = detector.detect(frame)

    frame = detector.draw(frame, results)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            frame = mustache_filter.apply(
                frame,
                face.landmark
            )

            frame = glasses_filter.apply(
                frame,
                face.landmark
            )

    cv2.imshow("Face Mesh", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("g"):
        glasses_filter.next_asset()

    if key == ord("q"):
        break


camera.release()