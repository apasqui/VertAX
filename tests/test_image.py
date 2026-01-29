"""Test the creation of a mesh from an image."""

from tifffile import imread

from vertax.pbc import PbcMesh

if __name__ == "__main__":
    img = imread("tests/test_image.tif")  # [:-101, :]  # non rect, odd and pair dimensions

    # mesh = PBCMesh.periodic_from_image(img)
    # mesh.save_mesh("image_mesh.npz")

    mesh = PbcMesh.load_mesh("image_mesh.npz")
    mesh.plot()
