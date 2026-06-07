import cv2

from camera_manager import CameraManager
from face_detector import FaceDetector
from filters.mustache_filter import MustacheFilter
from filters.glasses_filter import GlassesFilter
from filters.helmet_filter import HelmetFilter
from filters.welding_mask_filter import WeldingMaskFilter
from filters.hybridge_shield_filter import HybridgeShieldFilter
from filters.robot_mask_filter import RobotMaskFilter
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

hybridge_filter = HybridgeShieldFilter(
    "filtros-ar/models/hybridge_shield.glb"
)

robot_mask_filter = RobotMaskFilter(
    "filtros-ar/models/robot_mask.glb"
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
    "hybridge",
    hybridge_filter
)

pipeline.add(
    "robot_mask",
    robot_mask_filter
)

pipeline.add(
    "welding_mask",
    welding_mask_filter
)

pipeline.set_enabled("helmet", False)
pipeline.set_enabled("mustache", False)
pipeline.set_enabled("glasses", False)
pipeline.set_enabled("hybridge", False)
pipeline.set_enabled("welding_mask", False)
pipeline.set_enabled("robot_mask", False)


helmet_enabled = False
mustache_enabled = False
glasses_enabled = False
hybridge_enabled = False
robot_mask_enabled = False
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

    elif key == ord("4"):

        hybridge_enabled = not hybridge_enabled

        pipeline.set_enabled(
            "hybridge",
            hybridge_enabled
        )

        print(
            f"Hybridge Shield {'ON' if hybridge_enabled else 'OFF'}"
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

    elif key == ord("6"):

         robot_mask_enabled = (
            not robot_mask_enabled
        )
         
         pipeline.set_enabled(
            "robot_mask",
            robot_mask_enabled
        )
         
         print(
            f"Robot Mask {'ON' if robot_mask_enabled else 'OFF'}"
        )

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()