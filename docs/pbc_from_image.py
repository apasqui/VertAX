import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import an image as a base for the mesh

    With VertAX you can create a vertex model mesh from images !

    It will automatically create periodic boundary conditions for it.

    ## From a mask

    Suppose you have a mask that identifies the cells. It is of good quality and does not have holes :
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import tifffile

    mask_np_array = tifffile.imread("mask.tif")
    plt.imshow(mask_np_array)
    return mask_np_array, plt, tifffile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's make this mask periodic ! This can take up to a minute.
    """)
    return


@app.cell
def _(mask_np_array):
    from vertax import PbcMesh, plot_mesh
    _mesh = PbcMesh.periodic_from_mask(mask_np_array)
    plot_mesh(_mesh)
    return PbcMesh, plot_mesh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can recognize the original mask in the center of the mesh, which is two times bigger than the mask. A mirror operation makes the periodic boundary conditions.

    This mesh can then be used as in the inverse modelling example.

    ## From an image

    It is highly recommended to provide yourself a satisfying mask adapted to your situation. If you do not have such things, we propose a basic import from an image that we'll compute a mask. Note that this operation is generic and might not work for your image !

    It is as easy as the following :
    """)
    return


@app.cell
def _(plt, tifffile):
    img_array = tifffile.imread("image.tif")
    plt.imshow(img_array)
    return (img_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The previous mask was in fact computed by our method from this image, so the following result will be exactly the same:
    """)
    return


@app.cell
def _(PbcMesh, img_array, plot_mesh):
    _mesh = PbcMesh.periodic_from_image(img_array)  # will takes time too...
    plot_mesh(_mesh)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can recognize the original image in the center of the plotted mesh, which is two times bigger than the image. A mirror operation makes the periodic boundary conditions.

    This mesh can then be used as in the inverse modelling example.
    """)
    return


if __name__ == "__main__":
    app.run()
