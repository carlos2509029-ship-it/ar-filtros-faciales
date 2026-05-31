import cv2
import trimesh
import numpy as np


MODEL_PATH = (
    "filtros-ar/models/hybridge_shield.glb"
)


scene = trimesh.load(
    MODEL_PATH
)

mesh = list(
    scene.geometry.values()
)[0]

verts = mesh.vertices

faces = mesh.faces


canvas = np.zeros(
    (800, 800, 3),
    dtype=np.uint8
)


scale = 250

center_x = 400
center_y = 400


pts2d = []

for v in verts:

    x = int(
        center_x + v[0] * scale
    )

    y = int(
        center_y - v[1] * scale
    )

    pts2d.append(
        [x, y]
    )

pts2d = np.array(
    pts2d,
    dtype=np.int32
)

print("OpenCV OK")
print(cv2)
print(cv2.polylines)

print("Vertices:", len(verts))
print("Faces:", len(faces))
print("pts2d shape:", pts2d.shape)


for face in faces:

    poly = pts2d[face]

    cv2.polylines(
        canvas,
        [poly],
        True,
        (0, 255, 255),
        2
    )


cv2.imshow(
    "Hybridge Shield",
    canvas
)

cv2.waitKey(0)

cv2.destroyAllWindows()