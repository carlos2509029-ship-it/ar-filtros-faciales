import cv2
from pathlib import Path

from camera_manager import CameraManager
from face_detector import FaceDetector
from filters.mustache_filter import MustacheFilter


def main():
    cameras = CameraManager.list_available()

    if not cameras:
        print("No se detectaron cámaras disponibles.")
        return

    print("Cámaras detectadas:")
    for cam in cameras:
        print(f"  [{cam}] Cámara {cam}")

    choice = input("Selecciona cámara (Enter = primera): ").strip()
    selected_camera = cameras[0] if choice == "" else int(choice)

    if selected_camera not in cameras:
        print("La cámara seleccionada no está disponible.")
        return

    camera = CameraManager(selected_camera)
    detector = FaceDetector()

    filter_path = Path(__file__).resolve().parent / "assets" / "mustache.png"
    mustache_filter = MustacheFilter(str(filter_path))

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        results = detector.detect(frame)
        frame = detector.draw(frame, results)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            frame = mustache_filter.apply(frame, face_landmarks)

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()


if __name__ == "__main__":
    main()
