"""Cost functions collection, used for outer optimization.

A cost function can be any user-defined function but it has to respect a strict signature.

For a `PbcMesh` and `PbcBilevelOptimizer`, the cost function must have the following signature:
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
and return an Array (or float).

The names can vary and you can give default parameters. But the number and type of parameters is important.
You don't have to use every parameters but they all have to be here.
An unused parameters can of course also have the type None.

Same for `BoundedMesh` and `BoundedBilevelOptimizer`, but with a slightly different signature:
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vertTable_target: Array,
    angTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
and return an Array (or float).

Hopefully the variable names are self-explanatory.

You can create a function with this signature exactly that uses also locally-accessible external variable if you want.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.lax import fori_loop, stop_gradient
from jax.numpy import arange, array, diff, einsum, exp, expand_dims, int32, meshgrid, pi, sqrt, stack
from jax.numpy import sinc as npsinc
from jax.numpy.fft import ifft2
from numpy.typing import NDArray
from ott.geometry.geometry import Geometry
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from scipy.optimize import linear_sum_assignment

from vertax.geo import get_length

TARGET_RATIO = 2.0

# Image size, larger than the cube [-1,1]² --> FFT produces artefacts otherwise
s = array([-3, +3], dtype=float)
# Number of pixels along each axis
ns = array([256, 256], dtype=int32)
# Gaussian distribution parameters
## Mean value
mu = array([0, 0], dtype=float)
## Covariance
sigma = array([[5e-5, 0], [0, 5e-5]], dtype=float)


# Fourier grid defined by the Nyquist-Shannon sampling theorem
def _xigrid() -> Array:
    """Computes the Fourier grid.

      The grid is associated the regular grid defined in the box
    [-s₀/2,s₀/2]x[-s₁/2,s₁/2] and defined by the Nyquist-Shannon sampling theorem.
      * s is the box size (centered at 0).
      * ns is an integer vector defining the size of the grid.
    Returns the Fourier grid.
    """
    return stack(
        meshgrid(
            pi / s[0] * (2 * arange(ns[0], dtype=float) - ns[0] + 1),
            pi / s[1] * (2 * arange(ns[1], dtype=float) - ns[1] + 1),
            indexing="ij",
        )
    )


# FFT phase
def _fft_phase() -> Array:
    """Defines the phase term for the FFT computation.

      * ns is an integer vector defining the size of the grid.
    Returns the phase term.
    """
    return stack(
        meshgrid(
            exp(1j * pi * (1 - ns[0]) / ns[0] * arange(ns[0], dtype=float)),
            exp(1j * pi * (1 - ns[1]) / ns[1] * arange(ns[1], dtype=float)),
            indexing="ij",
        )
    ).prod(axis=0)


# Fourier transform of the Gaussian distribution
def _fourier_gaussian(mu: Array, sigma: Array, xi: Array) -> Array:
    """Computes the Fourier transform of the Gaussian distribution.

    With mean value μ and covariance matrix Σ (positive semi-definite)
    at points ξ.
    * μ is the mean value (vector of size (2,)).
    * Σ is the 2x2 covariance matrix (positive semi-definite).
    * ξ is an array of size (2,p₀,p₁,p₂,...)

    Returns:
    * The Fourier transform of the Gaussian distribution at points ξ.
        It has the size (p₀,p₁,p₂,...).
    """
    return exp(-1j * (expand_dims(mu, tuple(arange(1, len(xi.shape), dtype=int32))) * xi).sum(axis=0)) * exp(
        -_xisigmaxi(sigma, xi) / 2
    )


# Compute ξᵀΣξ
def _xisigmaxi(sigma: Array, xi: Array) -> Array:
    """Computes ξᵀΣξ for a matrix Σ and vectors ξ.

    * Σ is the 2x2 matrix.
    * ξ is an array of size (2,p₀,p₁,p₂,...)

    Returns:
    * ξᵀΣξ of size (p₀,p₁,p₂,...).
    """
    return (xi * einsum("ij,j...->i...", sigma, xi)).sum(axis=0)


# Pre-compute xi, xiexpa, phase, fac, fg
xi, phase = _xigrid(), _fft_phase()
xiexpa = expand_dims(xi, tuple(arange(1, 2)))
fac = ns.prod() / s.prod() * exp(1j * pi / 2 * (((ns - 1) ** 2 / ns).sum()))
fg = _fourier_gaussian(mu, sigma, xi)


###################################
# Change numpy definition of sinc #
###################################


@jit
def _sinc(x: Array) -> Array:
    """Sinus cardinal."""
    return npsinc(x / pi)


###############################################
# Fourier Transform Î of a line segment [a,b] #
#  Î(ξ) = |b-a| exp(-iξ(a+b)/2) sinc((b-a)/2) #
###############################################


# FT of all line segments (sum) with precomputed xi
@jit
def _sum_line_segment_fourier_transform(x: Array) -> Array:
    """Computes the Fourier transform of a set of line segments.

    * x is the array of line segments, the size is ((2,2,m)).
        x[:,:.i] is the line segment defined by the points
        (x[0,0,i],x[0,1,i]) and (x[1,0,i],x[1,1,i])
    * Needs a precomputed ξ, array of points where to compute
        the Fourier transform. The size is ((2,n)).
        The first dimension is for the Fourier vector coordinates.

    Returns:
    * The Fourier transform of the set of line segments at points ξ.
        It has the size (n,n).
    """
    # midx,difx=x.sum(axis=0)/2,diff(x,axis=0)[0]
    # return (sqrt((difx**2).sum(axis=0,keepdims=True)).T*exp(-1j*(midx.T@xi))*sinc((difx.T@xi)/2)).sum(axis=0)
    ####
    midx, difx = x.sum(axis=0) / 2, diff(x, axis=0)[0]
    midx, difx = expand_dims(midx, (-1, -2)), expand_dims(difx, (-1, -2))
    nonodifx = sqrt((difx**2).sum(axis=0))    # 1            no weight
    # nonodifx=1/sqrt((difx**2).sum(axis=0))  # (1/x)**2     weight
    # nonodifx*=exp(-nonodifx)                # exp(-x)      weight
    # nonodifx=(difx**2).sum(axis=0)          # x            weight
    # nonodifx=((difx**2).sum(axis=0))**2     # x**2         weight
    # nonodifx*=exp(nonodifx)                 # exp(-x)      weight
    return (nonodifx * exp(-1j * (midx * xiexpa).sum(axis=0)) * _sinc((xiexpa * difx).sum(axis=0) / 2)).sum(axis=0)
    ####


@jit
def _gaussian_blur_line_segments(x: Array) -> Array:
    """Blurs line segments.

    The line segments are defined in the box [-s₀/2,s₀/2]x[-s₁/2,s₁/2] at regularly spaced points
    with the Gaussian distribution of mean value μ and covariance matrix Σ (positive semi-definite).
    * x is the array of line segments: size ((2,2,m₀,m₁,m₂,...))

    Returns:
    * The blurring of the line segments with the Gaussian distribution.
    """
    return fac * ifft2(fg * _sum_line_segment_fourier_transform(x) * phase) * phase


##################
# COST FUNCTIONS #
##################


@partial(jit, static_argnums=(3, 4))
def cost_v2v(
    vertTable: Array,
    _heTable: Array,
    _faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    _heTable_target: Array,
    _faceTable_target: Array,
    selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    r"""Vertex-to-vertex cost with periodic boundary conditions (PBC).

    Implements the mean-squared-error between the current vertex positions
    :math:`\mathbf{x}_k` and the target positions :math:`\mathbf{x}_{\mathrm{GT},k}`:

    .. math::
        \mathcal{C}(\mathbf{X}, \mathbf{X}_{\mathrm{GT}})
            \;=\; \frac{1}{2N} \sum_{k=1}^{N}
                  \bigl\lVert \mathbf{x}_k - \mathbf{x}_{\mathrm{GT},k} \bigr\rVert^2,

    where the norm is the *minimum-image* distance over the 3x3 tiling of
    the periodic cell (``width`` x ``height``).
    """
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])

    x = vertTable[selected_verts]           # (N, 2)
    x_gt = vertTable_target[selected_verts] # (N, 2)

    # Nine periodic image offsets: {-1, 0, +1} along each axis.
    shifts = jnp.array(
        [[dx * width, dy * height] for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
        dtype=x.dtype,
    )                                       # (9, 2)

    # Pairwise differences against every periodic image of the target.
    deltas = x[:, None, :] - (x_gt[:, None, :] + shifts[None, :, :])  # (N, 9, 2)

    # Squared Euclidean distance per image, then minimum-image selection.
    sq_per_image = jnp.sum(deltas * deltas, axis=-1)                  # (N, 9)
    sq_min = jnp.min(sq_per_image, axis=-1)                           # (N,)

    return 0.5 * jnp.mean(sq_min)


# Default weights for ``cost_herve`` (see docstring).
_HERVE_LAMBDA_1 = 1.0  # shrink contacts absent from the target
_HERVE_LAMBDA_2 = 1.0  # match lengths on target contacts


def _vertex_weighted_adjacency(
    heTable: Array,
    vertTable: Array,
    faceTable: Array,
    width: float,
    height: float,
) -> tuple[Array, Array]:
    """Build symmetric vertex adjacency ``(bar_A, A)`` from undirected mesh edges.

    ``bar_A`` is binary; ``A`` stores edge lengths on adjacent pairs (zero otherwise).
    Each interior edge is counted once via the ``he < twin`` convention.
    """
    n_verts = vertTable.shape[0]
    n_hes = heTable.shape[0]
    he_ids = jnp.arange(n_hes, dtype=jnp.int32)
    twin = heTable[:, 2].astype(jnp.int32)
    face = heTable[:, 5]
    face_twin = face[twin]

    src = heTable[:, 3].astype(jnp.int32)
    tgt = heTable[:, 4].astype(jnp.int32)
    mask = (he_ids < twin) & (face != face_twin)

    lengths = vmap(
        lambda he: get_length(he, vertTable, heTable, faceTable, width, height),
        in_axes=0,
    )(he_ids)
    edge_len = jnp.where(mask, lengths, 0.0)

    bar_a = jnp.zeros((n_verts, n_verts))
    weighted = jnp.zeros((n_verts, n_verts))
    bar_a = bar_a.at[src, tgt].add(mask.astype(jnp.float32))
    bar_a = bar_a.at[tgt, src].add(mask.astype(jnp.float32))
    bar_a = jnp.clip(bar_a, 0.0, 1.0)

    weighted = weighted.at[src, tgt].add(edge_len)
    weighted = weighted.at[tgt, src].add(edge_len)
    return bar_a, weighted


@partial(jit, static_argnums=(3, 4))
def cost_herve(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    r"""Topological edge-length cost (Hervé's formulation).

    With binary adjacency :math:`\bar{A}_{\alpha\beta}` and weighted adjacency
    :math:`A_{\alpha\beta}(\mathbf{X}) = \ell_{\alpha\beta}(\mathbf{X})\,\bar{A}_{\alpha\beta}`:

    .. math::
        C_{\mathrm{topo}}
            = \lambda_1 \sum_{\alpha<\beta} A_{\alpha\beta}(\mathbf{X})
              \bigl(1 - \bar{A}_{\alpha\beta}^{\mathrm{target}}\bigr)
            + \lambda_2 \sum_{\alpha<\beta} \bar{A}_{\alpha\beta}^{\mathrm{target}}
              \bigl[A_{\alpha\beta}(\mathbf{X}) - A_{\alpha\beta}^{\mathrm{target}}\bigr]^2.

    Defaults: ``lambda_1 = lambda_2 = 1`` (pure topological minimization).
    For ``C = C_{\mathrm{geom}} + C_{\mathrm{topo}}`` with ``lambda_2 = 0``, combine
    with ``cost_v2v`` externally.
    """
    bar_curr, a_curr = _vertex_weighted_adjacency(
        heTable, vertTable, faceTable, width, height
    )
    bar_tgt, a_tgt = _vertex_weighted_adjacency(
        heTable_target, vertTable_target, faceTable_target, width, height
    )

    n_verts = vertTable.shape[0]
    idx_i, idx_j = jnp.triu_indices(n_verts, k=1)

    bar_c = bar_curr[idx_i, idx_j]
    bar_t = bar_tgt[idx_i, idx_j]
    a_c = a_curr[idx_i, idx_j]
    a_t = a_tgt[idx_i, idx_j]

    shrink_wrong = _HERVE_LAMBDA_1 * a_c * (1.0 - bar_t)
    preserve_target = _HERVE_LAMBDA_2 * bar_t * (a_c - a_t) ** 2
    return jnp.sum(shrink_wrong + preserve_target)


@partial(jit, static_argnums=(3, 4))
def cost_IAS(  # noqa: C901, N802
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    _width: float,
    _height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    r"""Differentiable Index Aware Structural Loss. Force to respect the topology.

    C_{IAS}(i,j)   = \sqrt{\sum_{k=1}^{N} (S_1(i,k) - S_2(j,k))^2}
    """
    L_box = jnp.sqrt(len(faceTable))

    def l2(x: Array, y: Array) -> Array:
        diff = x[:, None, :] - y[None, :, :]
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    def mse(x: Array, y: Array) -> Array:
        diff = x[:, None, :] - y[None, :, :]
        return jnp.sum(diff**2, axis=-1) + 1e-12

    def sinkhorn(a: Array, b: Array, C: Array, eps: float = 5e-2, n_iters: int = 50) -> Array:
        K = jnp.exp(-C / eps)
        u = jnp.ones_like(a)
        v = jnp.ones_like(b)

        def body(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            u, v = state
            u = a / (K @ v + 1e-12)
            v = b / (K.T @ u + 1e-12)
            return (u, v)

        u, v = fori_loop(0, n_iters, body, (u, v))
        return jnp.outer(u, v) * K

    # dual graph from half-edges
    def build_dual_adj(heTable: Array, n_faces: int) -> Array:
        face = heTable[:, 5]
        twin = heTable[:, 2]

        face_twin = face[twin]

        # mask: 1.0 for valid edges, 0.0 for boundary/self
        mask = (face != face_twin).astype(jnp.float32)

        A = jnp.zeros((n_faces, n_faces))

        # scatter with mask weighting (no boolean indexing)
        A = A.at[face, face_twin].add(mask)
        A = A.at[face_twin, face].add(mask)

        # binarize (avoid double counts)
        A = jnp.clip(A, 0.0, 1.0)

        return A

    # structure metric (1-hop + 2-hop)
    def structure_matrix(A: Array) -> Array:
        # A2 = A @ A
        # normalize to avoid scaling issues
        # return 2.0 - A - 0.5 * A2
        return 2.0 - A

    def get_face_vertices(
        he_start: Array, heTable: Array, vertTable: Array, max_edges: int = 20
    ) -> tuple[Array, Array]:
        """Collect vertices of one face using half-edge traversal.

        Returns fixed-size array (max_edges, 2) + mask
        """
        verts = jnp.zeros((max_edges, 2))
        mask = jnp.zeros((max_edges,))
        offset = jnp.array([0, 0])

        def body_fun(i: int, state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
            he, verts, mask, offset = state

            source = heTable[he, 3].astype(jnp.int32)
            off = heTable[he, 6:8]

            pos = vertTable[source] + offset * L_box  # jnp.array([width, height])

            verts = verts.at[i].set(pos)
            mask = mask.at[i].set(1.0)

            offset += off
            he_next = heTable[he, 1].astype(jnp.int32)

            return (he_next, verts, mask, offset)

        _he_final, verts, mask, offset = fori_loop(0, max_edges, body_fun, (he_start, verts, mask, offset))

        return verts, mask

    def polygon_centroid(verts: Array, mask: Array) -> Array:
        """Compute centroid from masked polygon vertices."""
        # shift for edges
        v = verts
        v_next = jnp.roll(v, -1, axis=0)

        cross = v[:, 0] * v_next[:, 1] - v_next[:, 0] * v[:, 1]
        cross = cross * mask

        area = jnp.sum(cross) / 2.0 + 1e-12

        cx = jnp.sum((v[:, 0] + v_next[:, 0]) * cross) / (6 * area)
        cy = jnp.sum((v[:, 1] + v_next[:, 1]) * cross) / (6 * area)

        return jnp.array([cx, cy])

    def compute_face_centroids(faceTable: Array, heTable: Array, vertTable: Array) -> Array:
        """Compute centroids for all faces."""
        he_start = faceTable[:].astype(jnp.int32)

        def single_face(he: Array) -> Array:
            verts, mask = get_face_vertices(he, heTable, vertTable)
            return polygon_centroid(verts, mask)

        return vmap(single_face)(he_start)

    alpha = 0.0
    """ATTENTION !!!!!!!!!!!!"""

    # face centroids
    X = compute_face_centroids(faceTable, heTable, vertTable)
    Y = compute_face_centroids(faceTable_target, heTable_target, vertTable_target)

    N = X.shape[0]
    M = Y.shape[0]

    # uniform weights
    a = jnp.ones(N) / N
    b = jnp.ones(M) / M

    # geometry cost
    C_geom = l2(X, Y)

    # structure cost
    A1 = build_dual_adj(heTable, N)
    A2 = build_dual_adj(heTable_target, M)

    S1 = structure_matrix(A1)
    S2 = structure_matrix(A2)

    # project structure into pairwise node cost
    # (cheap approximation of tem)
    # C_l2 = l2(S1, S2)
    C_mse = mse(S1, S2)

    # combined cost
    C_total = alpha * C_geom + (1.0 - alpha) * C_mse

    # transport
    gamma = sinkhorn(a, b, C_total)

    # final loss
    loss = jnp.sum(gamma * C_total)

    return loss


# ---------------------------------------------------------------------------
# Helpers used by `cost_IAS` (kept at module level so they JIT-compile once
# and so the cost function itself stays short and readable).
# ---------------------------------------------------------------------------

# Sinkhorn solver matching the original hand-rolled implementation:
# kernel-mode (lse_mode=False), exactly 50 fixed iterations, eps = 5e-2.
_IAS_SINKHORN = Sinkhorn(
    lse_mode=False,
    min_iterations=50,
    max_iterations=50,
    threshold=0.0,
)
_IAS_EPSILON = 5e-2


def _pairwise_sq(x: Array, y: Array) -> Array:
    """Pairwise squared L2 distance with a small epsilon for differentiability."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sum(diff**2, axis=-1) + 1e-12


def _pairwise_l2(x: Array, y: Array) -> Array:
    """Pairwise L2 distance with a small epsilon for differentiability."""
    return jnp.sqrt(_pairwise_sq(x, y))


def _build_dual_adj(heTable: Array, n_faces: int) -> Array:
    """Symmetric, binarized dual-graph adjacency between faces (via half-edge twins)."""
    face = heTable[:, 5]
    twin = heTable[:, 2]
    face_twin = face[twin]

    # 1.0 for genuine inter-face edges, 0.0 for boundary/self loops.
    mask = (face != face_twin).astype(jnp.float32)

    A = jnp.zeros((n_faces, n_faces))
    A = A.at[face, face_twin].add(mask)
    A = A.at[face_twin, face].add(mask)
    return jnp.clip(A, 0.0, 1.0)


def _structure_matrix(A: Array) -> Array:
    """Structural-distance proxy combining 1-hop and 2-hop dual adjacency."""
    return 2.0 - A - 0.5 * (A @ A)


def _face_vertices(
    he_start: Array,
    heTable: Array,
    vertTable: Array,
    L_box: Array,
    max_edges: int = 20,
) -> tuple[Array, Array]:
    """Collect (up to ``max_edges``) vertices of a face via half-edge traversal.

    Returns:
        verts: ``(max_edges, 2)`` fixed-size vertex array (padded with zeros).
        mask:  ``(max_edges,)`` mask marking valid (non-padded) entries.
    """
    verts = jnp.zeros((max_edges, 2))
    mask = jnp.zeros((max_edges,))
    offset = jnp.array([0, 0])

    def body(i: int, state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        he, verts, mask, offset = state
        source = heTable[he, 3].astype(jnp.int32)
        off = heTable[he, 6:8]
        pos = vertTable[source] + offset * L_box
        verts = verts.at[i].set(pos)
        mask = mask.at[i].set(1.0)
        offset = offset + off
        he_next = heTable[he, 1].astype(jnp.int32)
        return (he_next, verts, mask, offset)

    _, verts, mask, _ = fori_loop(0, max_edges, body, (he_start, verts, mask, offset))
    return verts, mask


def _polygon_centroid(verts: Array, mask: Array) -> Array:
    """Shoelace-formula centroid of a (possibly partially-masked) polygon."""
    v = verts
    v_next = jnp.roll(v, -1, axis=0)
    cross = (v[:, 0] * v_next[:, 1] - v_next[:, 0] * v[:, 1]) * mask

    area = jnp.sum(cross) / 2.0 + 1e-12
    cx = jnp.sum((v[:, 0] + v_next[:, 0]) * cross) / (6.0 * area)
    cy = jnp.sum((v[:, 1] + v_next[:, 1]) * cross) / (6.0 * area)
    return jnp.array([cx, cy])


def _face_centroids(faceTable: Array, heTable: Array, vertTable: Array, L_box: Array) -> Array:
    """Centroid of every face of the mesh."""
    he_start = faceTable[:].astype(jnp.int32)

    def one(he: Array) -> Array:
        verts, mask = _face_vertices(he, heTable, vertTable, L_box)
        return _polygon_centroid(verts, mask)

    return vmap(one)(he_start)


@partial(jit, static_argnums=(3, 4))
def cost_IAS_ott_jax(  # noqa: N802
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    _width: float,
    _height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    r"""Differentiable Index-Aware Structural (IAS) loss.

    Penalizes topological mismatch between the current and target meshes via
    entropy-regularized optimal transport over face centroids.

    Pipeline:
        1. Compute face centroids of both meshes.
        2. Build the half-edge dual-graph adjacency of each mesh.
        3. Derive a structural metric per face from 1-hop + 2-hop adjacency.
        4. Solve OT on the structural pairwise costs.
        5. Return the transport-weighted combined cost

           .. math::
               C_{ij} = \alpha \, \lVert X_i - Y_j \rVert
                      + (1-\alpha) \, \lVert S_1(i,:) - S_2(j,:) \rVert^2,

           with ``gamma`` stop-gradient'd so vertex updates follow the
           geometric term only (ott-jax Sinkhorn backward is unstable here).
    """
    alpha = 0.5

    L_box = jnp.sqrt(len(faceTable))

    X = _face_centroids(faceTable, heTable, vertTable, L_box)
    Y = _face_centroids(faceTable_target, heTable_target, vertTable_target, L_box)
    N, M = X.shape[0], Y.shape[0]

    C_geom = _pairwise_l2(X, Y)

    S1 = _structure_matrix(_build_dual_adj(heTable, N))
    S2 = _structure_matrix(_build_dual_adj(heTable_target, M))
    C_struct = _pairwise_sq(S1, S2)

    C_total = alpha * C_geom + (1.0 - alpha) * C_struct

    a = jnp.ones(N) / N
    b = jnp.ones(M) / M

    geom = Geometry(cost_matrix=C_struct, epsilon=_IAS_EPSILON)
    gamma = _IAS_SINKHORN(LinearProblem(geom, a=a, b=b)).matrix
    gamma = stop_gradient(gamma)

    return jnp.sum(gamma * C_total)


@partial(jit, static_argnums=(3, 4))
def cost_IAS_ott_jax_move_vertices(  # noqa: N802
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    _width: float,
    _height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    r"""IAS loss with topology-driven matching and a geometric vertex gradient.

    Same OT matching as ``cost_IAS``, but ``gamma`` is stop-gradient'd and only
    the transport-weighted squared centroid distances contribute to the loss.
    """
    L_box = jnp.sqrt(len(faceTable))

    X = _face_centroids(faceTable, heTable, vertTable, L_box)
    Y = _face_centroids(faceTable_target, heTable_target, vertTable_target, L_box)
    N, M = X.shape[0], Y.shape[0]

    S1 = _structure_matrix(_build_dual_adj(heTable, N))
    S2 = _structure_matrix(_build_dual_adj(heTable_target, M))
    C_struct = _pairwise_sq(S1, S2)

    a = jnp.ones(N) / N
    b = jnp.ones(M) / M

    geom = Geometry(cost_matrix=C_struct, epsilon=_IAS_EPSILON)
    gamma = _IAS_SINKHORN(LinearProblem(geom, a=a, b=b)).matrix
    gamma = stop_gradient(gamma)

    C_geom_sq = _pairwise_sq(X, Y)

    return jnp.sum(gamma * C_geom_sq)


def _main() -> None:
    """Quick sanity check for ``cost_v2v``, ``cost_IAS`` and ``cost_IAS_move_vertices``.

    Pipeline:
        1. Build an out-of-equilibrium ``PbcMesh`` from random Voronoi seeds.
        2. Bring it to mechanical equilibrium with ``energy_shape_factor_hetero``
           and shape factors drawn from N(3.9, 0.1), allowing T1 transitions.
        3. Build a target mesh as a copy of the initial out-of-equilibrium
           configuration with shape factors resampled from the same N(3.9, 0.1)
           distribution, and bring it to equilibrium.
        4. For each cost function, compute d cost / d vertTable and perform 10
           gradient-descent steps on the vertices of the (equilibrated) initial
           mesh, applying ``update_T1`` after each step (T1 accepted when the
           outer cost decreases).
    """
    import jax
    import optax

    from pathlib import Path

    from vertax.bilevelopt.bilevelopt import _apply_perm_to_state, _build_t1_repair_perm
    from vertax.energy import energy_shape_factor_hetero
    from vertax.meshes.pbc_mesh import PbcMesh
    from vertax.meshes.plot import plot_mesh
    from vertax.opt import inner_opt
    from vertax.topo import update_T1

    plot_root = Path("logs/cost_move_vertices_plots")
    plot_root.mkdir(parents=True, exist_ok=True)

    def _mesh_from_state(vt: Array, ht: Array, ft: Array) -> PbcMesh:
        mesh = PbcMesh.copy_mesh(init_mesh)
        mesh.vertices = vt
        mesh.edges = ht
        mesh.faces = ft
        return mesh

    def _save_configuration_plots(cost_name: str, vt_final: Array, ht_final: Array, ft_final: Array) -> None:
        plot_dir = plot_root / cost_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        configs = (
            ("initial", vt_init_eq, ht_init_eq, ft_init_eq),
            ("target", vt_tgt, ht_tgt, ft_tgt),
            ("final", vt_final, ht_final, ft_final),
        )
        for label, vt, ht, ft in configs:
            plot_mesh(
                _mesh_from_state(vt, ht, ft),
                show=False,
                save=True,
                save_path=str(plot_dir / f"{label}.png"),
                title=f"{cost_name}: {label}",
            )
        print(f"  plots saved to {plot_dir}/")

    # ------------------------------------------------------------------
    # 1. Initial out-of-equilibrium mesh from random seeds
    # ------------------------------------------------------------------
    n_cells = 20
    width = float(jnp.sqrt(n_cells))
    height = width
    init_mesh = PbcMesh.from_random_seeds(
        nb_seeds=n_cells, width=width, height=height, random_key=42
    )
    vt_ooe = init_mesh.vertices
    ht_ooe = init_mesh.edges
    ft_ooe = init_mesh.faces

    n_verts = init_mesh.nb_vertices
    n_hes = init_mesh.nb_half_edges
    n_faces = init_mesh.nb_faces
    print(
        f"Initial mesh: {n_faces} faces, {n_hes // 2} edges, {n_verts} vertices, "
        f"box = {width:.3f} x {height:.3f}"
    )

    # ------------------------------------------------------------------
    # 2. Inner-loop energy: heterogeneous shape factor on all faces
    # ------------------------------------------------------------------
    def energy_fn(
        vertTable: Array,
        heTable: Array,
        faceTable: Array,
        _vert_params: Array,
        _he_params: Array,
        face_params: Array,
    ) -> Array:
        selected_faces = jnp.arange(len(faceTable))
        return energy_shape_factor_hetero(
            vertTable, heTable, faceTable, width, height, selected_faces, face_params
        )

    vert_params = jnp.zeros((n_verts,))
    he_params = jnp.zeros((n_hes,))

    key_init = jax.random.PRNGKey(0)
    sf_init = 3.9 + 0.1 * jax.random.normal(key_init, shape=(n_faces,))

    solver = optax.sgd(learning_rate=0.01)

    print("Inner loop on initial mesh -> equilibrium ...")
    (vt_init_eq, ht_init_eq, ft_init_eq), L_hist_init = inner_opt(
        vt_ooe,
        ht_ooe,
        ft_ooe,
        width,
        height,
        vert_params,
        he_params,
        sf_init,
        L_in=energy_fn,
        solver=solver,
        min_dist_T1=0.005,
        iterations_max=500,
        tolerance=1e-4,
        patience=5,
        update_t1_func=update_T1,
    )
    print(f"  energy: {float(L_hist_init[0]):.4f} -> {float(L_hist_init[-1]):.4f}")

    # ------------------------------------------------------------------
    # 3. Target mesh: copy of the initial OOE configuration, new shape factors
    # ------------------------------------------------------------------
    key_target = jax.random.PRNGKey(123)
    sf_target = 3.9 + 0.1 * jax.random.normal(key_target, shape=(n_faces,))

    print("Inner loop on target mesh -> equilibrium ...")
    (vt_tgt, ht_tgt, ft_tgt), L_hist_tgt = inner_opt(
        vt_ooe,
        ht_ooe,
        ft_ooe,
        width,
        height,
        vert_params,
        he_params,
        sf_target,
        L_in=energy_fn,
        solver=solver,
        min_dist_T1=0.005,
        iterations_max=500,
        tolerance=1e-4,
        patience=5,
        update_t1_func=update_T1,
    )
    print(f"  energy: {float(L_hist_tgt[0]):.4f} -> {float(L_hist_tgt[-1]):.4f}")

    # ------------------------------------------------------------------
    # 4. Outer loop: gradient descent on each cost function w.r.t. vertTable.
    # ------------------------------------------------------------------
    cost_fns = {
        "cost_v2v": cost_v2v,
        "cost_IAS": cost_IAS,
        "cost_IAS_ott_jax": cost_IAS_ott_jax,
        "cost_IAS_ott_jax_move_vertices": cost_IAS_ott_jax_move_vertices,
        "cost_herve": cost_herve,
    }

    outer_lr = 0.05
    n_outer_steps = 1000
    min_dist_T1 = 0.005
    selected_verts = jnp.arange(n_verts)
    selected_hes = jnp.arange(n_hes)
    selected_faces = jnp.arange(n_faces)

    for name, cost_fn in cost_fns.items():
        print(f"\n--- Gradient descent on {name} ({n_outer_steps} steps) ---")

        grad_cost_fn = jax.jit(jax.grad(cost_fn, argnums=0), static_argnums=(3, 4))

        def cost_for_t1(
            vertTable: Array,
            heTable: Array,
            faceTable: Array,
            _vert_params: Array,
            _he_params: Array,
            _face_params: Array,
        ) -> Array:
            return cost_fn(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vt_tgt,
                ht_tgt,
                ft_tgt,
            )

        vt = vt_init_eq
        ht = ht_init_eq
        ft = ft_init_eq

        c0 = float(cost_fn(vt, ht, ft, width, height, vt_tgt, ht_tgt, ft_tgt))
        print(f"  step  0  cost = {c0:.6f}")

        for step in range(1, n_outer_steps + 1):
            g_vt = grad_cost_fn(vt, ht, ft, width, height, vt_tgt, ht_tgt, ft_tgt)
            vt = vt - outer_lr * g_vt
            ht_before = ht
            vt, ht, ft = update_T1(
                vt,
                ht,
                ft,
                width,
                height,
                vert_params,
                he_params,
                sf_init,
                cost_for_t1,
                min_dist_T1,
                selected_verts,
                selected_hes,
                selected_faces,
            )
            perm = _build_t1_repair_perm(vt, vt_tgt, ht_before, ht, width, height)
            vt, ht = _apply_perm_to_state(perm, vt, ht)
            c = float(cost_fn(vt, ht, ft, width, height, vt_tgt, ht_tgt, ft_tgt))
            if step % 100 == 0:
                print(f"  step {step:2d}  cost = {c:.6f}")

        _save_configuration_plots(name, vt, ht, ft)


if __name__ == "__main__":
    _main()
