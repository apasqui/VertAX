import jax.numpy as jnp
from jax import Array, jit, vmap

from vertax.geo import get_area, get_length, get_perimeter


@jit
def cell_energy(face, face_param, vertTable, heTable, faceTable, width: float, height: float):
    area = get_area(face, vertTable, heTable, faceTable, width, height)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable, width, height)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)


@jit
def energy_shape_factor_homo(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    def mapped_fn(face, param):
        return cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable),) + face_params.shape[1:])
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
    return jnp.sum(cell_energies)


@jit
def energy_shape_factor_hetero(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    def mapped_fn(face, param):
        return cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    cell_energies = vmap(mapped_fn)(selected_faces, face_params[selected_faces])
    # cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params)
    return jnp.sum(cell_energies)


@jit
def area_part(
    face: float, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
):
    a = get_area(face, vertTable, heTable, faceTable, width, height)
    return (a - face_param) ** 2


@jit
def hedge_part(
    he: float, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
):
    length = get_length(he, vertTable, heTable, faceTable, width, height)
    return he_param * length


@jit
def energy_line_tensions(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    K_areas = 20

    def mapped_areas_part(face, face_param):
        return area_part(face, face_param, vertTable, heTable, faceTable, width, height)

    def mapped_hedges_part(he, he_param):
        return hedge_part(he, he_param, vertTable, heTable, faceTable, width, height)

    areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), face_params)
    hedges_part = vmap(mapped_hedges_part)(jnp.arange(len(heTable)), he_params)
    # areas_part = vmap(mapped_areas_part)(selected_faces, face_params[selected_faces])
    # hedges_part = vmap(mapped_hedges_part)(selected_hes, he_params[selected_hes])
    return jnp.sum(hedges_part) + (0.5 * K_areas) * jnp.sum(areas_part)
