import trimesh
import numpy as np

vertices = np.array([
    [-0.60,  0.80, 0.0],  # frente izq
    [ 0.60,  0.80, 0.0],  # frente der

    [-0.85,  0.20, 0.0],  # sien izq
    [ 0.85,  0.20, 0.0],  # sien der

    [-0.75, -0.50, 0.0],  # mejilla izq
    [ 0.75, -0.50, 0.0],  # mejilla der

    [ 0.00, -1.00, 0.0],  # barbilla

    [-0.20,  0.10, 0.2],  # nariz izq
    [ 0.20,  0.10, 0.2],  # nariz der
    [ 0.00, -0.20, 0.3],  # punta nariz
])

faces = np.array([
    [0, 1, 7],
    [1, 8, 7],

    [1, 3, 8],
    [0, 7, 2],

    [2, 7, 4],
    [3, 5, 8],

    [7, 8, 9],

    [4, 7, 6],
    [5, 6, 8],

    [4, 5, 6]
])

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces
)

mesh.export(
    "filtros-ar/models/robot_mask.glb"
)

print("Robot Mask exportado")