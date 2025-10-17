import jax
import jax.numpy as jnp
import optax
from jax import grad, jacfwd, jit, lax

from vertax.geo import update_pbc
from vertax.topo import update_T1

###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


def minimize(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    L_in,
    solver,
    min_dist_T1,
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    argnums=0,
):
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    iterations_max = int(iterations_max)
    patience = int(patience)

    # --- select the target array and indices for the optimizer ---
    if argnums == 0:
        x0, sel = vertTable, selected_verts
    elif argnums == 1:
        x0, sel = heTable, selected_hes
    elif argnums == 2:
        x0, sel = faceTable, selected_faces
    elif argnums == 3:
        x0, sel = vert_params, selected_verts
    elif argnums == 4:
        x0, sel = he_params, selected_hes
    elif argnums == 5:
        x0, sel = face_params, selected_faces
    else:
        raise ValueError("argnums must be in {0,1,2,3,4,5}")

    # --- check if the target is float-type before differentiating ---
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        raise TypeError(
            f"Cannot differentiate argnums={argnums}: "
            f"target array has dtype {x0.dtype}. "
            f"Use a float array (vertTable, vert_params, he_params, face_params) only."
        )

    opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = L_in(
        vertTable[selected_verts],
        heTable[selected_hes],
        faceTable[selected_faces],
        vert_params[selected_verts],
        he_params[selected_hes],
        face_params[selected_faces],
    )
    L_in_list = jnp.zeros((iterations_max,)).at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)

    @jit
    def update_step(carry):
        vt, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list = carry

        # 1) Compute loss
        L_current = L_in(
            vt[selected_verts],
            ht[selected_hes],
            ft[selected_faces],
            vp[selected_verts],
            hp[selected_hes],
            fp[selected_faces],
        )
        L_list = L_list.at[step_count].set(L_current)

        # 2) Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)
        stagnation_count = stagnation_count + jnp.where(rel_var < tolerance, 1, -stagnation_count)
        should_stop = (stagnation_count >= patience) | (step_count >= iterations_max - 1)

        # 3) Gradient wrt chosen argnums
        grads = jacfwd(L_in, argnums=argnums)(vt, ht, ft, vp, hp, fp)

        # 4) Optimizer update
        updates, opt_state = solver.update(grads, opt_state)

        # 5) Apply updates to the chosen array on selected indices
        updates_sel = updates.at[sel].get()
        if argnums == 0:
            vt = vt.at[sel].set(vt[sel] + updates_sel)
        elif argnums == 1:
            ht = ht.at[sel].set(ht[sel] + updates_sel)
        elif argnums == 2:
            ft = ft.at[sel].set(ft[sel] + updates_sel)
        elif argnums == 3:
            vp = vp.at[sel].set(vp[sel] + updates_sel)
        elif argnums == 4:
            hp = hp.at[sel].set(hp[sel] + updates_sel)
        elif argnums == 5:
            fp = fp.at[sel].set(fp[sel] + updates_sel)

        # 6) Geometry updates
        vt, ht, ft = update_pbc(vt, ht, ft)
        vt, ht, ft = update_T1(vt, ht, ft, vp, hp, fp, L_in, min_dist_T1)

        # 7) Shift prev_L_values
        prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        prev_L_values = prev_L_values.at[0].set(L_current)

        step_count += 1

        return (vt, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list)

    def cond_fn(state):
        return jnp.logical_not(state[10])  # should_stop

    final_state = lax.while_loop(
        cond_fn,
        update_step,
        (
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            opt_state,
            prev_L_values,
            stagnation_count,
            step_count,
            should_stop,
            L_in_list,
        ),
    )

    vt_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state
    final_L_list = L_hist[:step_f]

    return (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), final_L_list


def inner_opt(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    L_in,
    solver,
    min_dist_T1,
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
):
    # Use the general minimize function with argnums=0 (optimize vertTable)
    (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), L_hist = minimize(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver,
        min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        argnums=0,
    )

    # Return updated arrays and loss history
    return (vt_f, ht_f, ft_f), L_hist


def cost_ad(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    min_dist_T1,
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    image_target=None,
):
    (vertTable, heTable, faceTable), L_in = inner_opt(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
    )

    loss_out_value = L_out(
        vertTable,
        heTable,
        faceTable,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_out_value


def outer_opt(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    solver_outer,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
):
    grad_verts = jacfwd(cost_ad, argnums=3)(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    grad_hes = jacfwd(cost_ad, argnums=4)(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    grad_faces = jacfwd(cost_ad, argnums=5)(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]
    he_params = updated_params["he_params"]
    face_params = updated_params["face_params"]

    return vert_params, he_params, face_params


#############################
## EQUILIBTIUM PROPAGATION ##
#############################


def loss_ep(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
    beta,
):
    loss_inner_value = L_in(vertTable, heTable, faceTable, vert_params, he_params, face_params)

    loss_outer_value = L_out(
        vertTable,
        heTable,
        faceTable,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    loss_ep_value = loss_inner_value + (beta * loss_outer_value)

    return loss_ep_value


def forward(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
    beta,
):
    def loss_ep_forward(vt, ht, ft, vp, hp, fp):
        return loss_ep(
            vt,
            ht,
            ft,
            vp,
            hp,
            fp,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
            beta,
        )

    (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), final_L_list = minimize(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in=loss_ep_forward,
        solver=solver_inner,
        min_dist_T1=min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        argnums=0,
    )

    return (vt_f, ht_f, ft_f), final_L_list


def outer_eq_prop(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    solver_outer,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
    beta,
):
    (vertTable_free, heTable_free, faceTable_free), loss_free = forward(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=0.0,
    )

    (vertTable_nudged, heTable_nudged, faceTable_nudged), loss_nudged = forward(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_loss_ep_free_verts = jacfwd(loss_ep, argnums=3)(
        vertTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=0.0,
    )

    grad_loss_ep_nudged_verts = jacfwd(loss_ep, argnums=3)(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_loss_ep_free_hes = jacfwd(loss_ep, argnums=4)(
        vertTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=0.0,
    )

    grad_loss_ep_nudged_hes = jacfwd(loss_ep, argnums=4)(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_loss_ep_free_faces = jacfwd(loss_ep, argnums=5)(
        vertTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=0.0,
    )

    grad_loss_ep_nudged_faces = jacfwd(loss_ep, argnums=5)(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_verts = (1.0 / beta) * ((grad_loss_ep_nudged_verts) - (grad_loss_ep_free_verts))
    grad_hes = (1.0 / beta) * ((grad_loss_ep_nudged_hes) - (grad_loss_ep_free_hes))
    grad_faces = (1.0 / beta) * ((grad_loss_ep_nudged_faces) - (grad_loss_ep_free_faces))

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]
    he_params = updated_params["he_params"]
    face_params = updated_params["face_params"]

    return vert_params, he_params, face_params


###########################
## IMPLICIT DIFFERENTION ##
###########################


def outer_implicit(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    solver_outer,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
):
    def L_in_flatten(vertTable_flatten, heTable, faceTable, vert_params, he_params, face_params):
        vertTable_tmp = jnp.hstack(
            (vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2), jnp.zeros((len(vertTable_flatten) // 2, 1)))
        )
        return L_in(vertTable_tmp, heTable, faceTable, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), L_in_value = inner_opt(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
    )

    H = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    crossderivative_verts = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=3)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=4)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=5)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    L_in_derivative_verts = -jax.numpy.linalg.solve(H, crossderivative_verts)
    L_in_derivative_hes = -jax.numpy.linalg.solve(H, crossderivative_hes)
    L_in_derivative_faces = -jax.numpy.linalg.solve(H, crossderivative_faces)

    grad_verts = (
        L_in_derivative_verts.T
        @ grad(L_out, argnums=0)(
            vertTable_eq, heTable_eq, faceTable_eq, vertTable_target, heTable_target, faceTable_target, image_target
        )[:, :2].flatten()
    )
    grad_hes = (
        L_in_derivative_hes.T
        @ grad(L_out, argnums=0)(
            vertTable_eq, heTable_eq, faceTable_eq, vertTable_target, heTable_target, faceTable_target, image_target
        )[:, :2].flatten()
    )
    grad_faces = (
        L_in_derivative_faces.T
        @ grad(L_out, argnums=0)(
            vertTable_eq, heTable_eq, faceTable_eq, vertTable_target, heTable_target, faceTable_target, image_target
        )[:, :2].flatten()
    )

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]
    he_params = updated_params["he_params"]
    face_params = updated_params["face_params"]

    return vert_params, he_params, face_params


##########################
## ADJOINT STATE METHOD ##
##########################


def outer_adjoint_state(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    solver_outer,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
):
    def L_in_flatten(vertTable_flatten, heTable, faceTable, vert_params, he_params, face_params):
        vertTable_tmp = jnp.hstack(
            (vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2), jnp.zeros((len(vertTable_flatten) // 2, 1)))
        )
        return L_in(vertTable_tmp, heTable, faceTable, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), L_in_value = inner_opt(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
    )

    H = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    # print(jax.numpy.linalg.eig(H)[0].max())
    # print(jax.numpy.linalg.eig(H)[0].min())
    # print('\n')

    crossderivative_verts = jacfwd(jacfwd(L_in_flatten, argnums=3), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = jacfwd(jacfwd(L_in_flatten, argnums=4), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = jacfwd(jacfwd(L_in_flatten, argnums=5), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    gradout = grad(L_out, argnums=0)(
        vertTable_eq, heTable_eq, faceTable_eq, vertTable_target, heTable_target, faceTable_target, image_target
    )[:, :2].flatten()

    Lambda = -jax.numpy.linalg.solve(H, gradout)

    grad_verts = crossderivative_verts @ Lambda
    grad_hes = crossderivative_hes @ Lambda
    grad_faces = crossderivative_faces @ Lambda

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]
    he_params = updated_params["he_params"]
    face_params = updated_params["face_params"]

    return vert_params, he_params, face_params


#############
## WRAPPER ##
#############


def bilevel_opt(
    vertTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    solver_outer,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target=None,
    beta=None,
    method="ad",
):
    if method == "ad":
        vert_params, he_params, face_params = outer_opt(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            solver_inner,
            solver_outer,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
        )

        (vertTable, heTable, faceTable), cost = inner_opt(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            L_in,
            solver_inner,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
        )

    elif method == "ep":
        vert_params, he_params, face_params = outer_eq_prop(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            solver_inner,
            solver_outer,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
            beta,
        )

        (vertTable, heTable, faceTable), cost = forward(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            solver_inner,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
            beta=0.0,
        )

    elif method == "id":
        vert_params, he_params, face_params = outer_implicit(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            solver_inner,
            solver_outer,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
        )

        (vertTable, heTable, faceTable), cost = inner_opt(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            L_in,
            solver_inner,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
        )

    elif method == "as":
        vert_params, he_params, face_params = outer_adjoint_state(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            solver_inner,
            solver_outer,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
        )

        (vertTable, heTable, faceTable), cost = inner_opt(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            L_in,
            solver_inner,
            min_dist_T1,
            iterations_max,
            tolerance,
            patience,
            selected_verts,
            selected_hes,
            selected_faces,
        )

    return (vertTable, heTable, faceTable, vert_params, he_params, face_params), cost
