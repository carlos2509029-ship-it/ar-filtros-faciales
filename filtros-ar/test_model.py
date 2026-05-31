import trimesh


MODEL_PATH = (
    "filtros-ar/models/hybridge_shield.glb"
)


model = trimesh.load(
    MODEL_PATH
)

print()
print("TIPO:")
print(type(model))
print()

if isinstance(
    model,
    trimesh.Scene
):
    print("Es una Scene")
    print(
        "Geometrías:",
        len(model.geometry)
    )

    for name, geo in model.geometry.items():

        print()
        print("Objeto:", name)

        print(
            "Vertices:",
            len(geo.vertices)
        )

        print(
            "Caras:",
            len(geo.faces)
        )

else:

    print("Es un Trimesh")

    print(
        "Vertices:",
        len(model.vertices)
    )

    print(
        "Caras:",
        len(model.faces)
    )