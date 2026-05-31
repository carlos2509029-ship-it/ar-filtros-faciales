import cv2

from camera_manager import CameraManager
from face_detector import FaceDetector

from filters.mustache_filter import MustacheFilter
from filters.glasses_filter import GlassesFilter
from filters.helmet_filter import HelmetFilter
from filters.welding_mask_filter import WeldingMaskFilter

from filter_pipeline import FilterPipeline


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
        "filtros-ar/assets/glasses2.png",
        "filtros-ar/assets/safety_goggles.png"
    ]
)

helmet_filter = HelmetFilter(
    "filtros-ar/assets/mechanic_helmet.png"
)

welding_mask_filter = WeldingMaskFilter(
    "filtros-ar/assets/welding_mask.png"
)


pipeline = FilterPipeline()

pipeline.add(
    "helmet",
    helmet_filter
)

pipeline.add(
    "mustache",
    mustache_filter
)

pipeline.add(
    "glasses",
    glasses_filter
)

pipeline.add(
    "welding_mask",
    welding_mask_filter
)


helmet_enabled = True
mustache_enabled = True
glasses_enabled = True
welding_mask_enabled = False


while True:

    frame = camera.get_frame()

    if frame is None:
        break

    results = detector.detect(frame)

    frame = detector.draw(frame, results)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            frame = pipeline.apply(
                frame,
                face.landmark
            )

    cv2.imshow(
        "Face Mesh",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("g"):
        glasses_filter.next_asset()

    elif key == ord("1"):

        mustache_enabled = not mustache_enabled

        pipeline.set_enabled(
            "mustache",
            mustache_enabled
        )

        print(
            f"Mostacho {'ON' if mustache_enabled else 'OFF'}"
        )

    elif key == ord("2"):

        glasses_enabled = not glasses_enabled

        pipeline.set_enabled(
            "glasses",
            glasses_enabled
        )

        print(
            f"Lentes {'ON' if glasses_enabled else 'OFF'}"
        )

    elif key == ord("3"):

        helmet_enabled = not helmet_enabled

        pipeline.set_enabled(
            "helmet",
            helmet_enabled
        )

        print(
            f"Casco {'ON' if helmet_enabled else 'OFF'}"
        )

    elif key == ord("5"):

        welding_mask_enabled = (
            not welding_mask_enabled
        )

        pipeline.set_enabled(
            "welding_mask",
            welding_mask_enabled
        )

        print(
            f"Mascara Soldador {'ON' if welding_mask_enabled else 'OFF'}"
        )

    elif key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()