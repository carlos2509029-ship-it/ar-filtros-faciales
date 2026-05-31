import trimesh
import numpy as np


vertices = np.array([
    [-0.40,  0.70, 0.0],
    [ 0.40,  0.70, 0.0],
    [ 0.70,  0.20, 0.0],
    [ 0.55, -0.50, 0.0],
    [ 0.00, -0.90, 0.0],
    [-0.55, -0.50, 0.0],
    [-0.70,  0.20, 0.0],
])

faces = np.array([
    [0, 1, 6],
    [1, 2, 6],
    [2, 5, 6],
    [2, 3, 5],
    [3, 4, 5]
])

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces
)

print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.faces))

mesh.export(
    "filtros-ar/models/hybridge_shield.glb"
)

print("Escudo Hybridge exportado")