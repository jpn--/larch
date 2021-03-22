import numpy as np
from ..exceptions import BHHHSimpleStepFailure


def propose_direction(bhhh, dloglike, freedoms):
    direction = np.zeros_like(dloglike)
    try:
        ibhhh = np.linalg.inv(bhhh[freedoms, :][:, freedoms])
    except np.linalg.LinAlgError:
        ibhhh = np.linalg.pinv(bhhh[freedoms, :][:, freedoms])
    direction1 = np.dot(ibhhh, dloglike[freedoms])
    # direction1 = np.linalg.lstsq(
    #     bhhh[freedoms, :][:, freedoms],
    #     dloglike[freedoms],
    #     rcond=None,
    # )[0]
    direction[freedoms] = direction1
    return direction


def fit_bhhh(
        model,
        steplen=1.0,
        momentum=5,
        logger=None,
        ctol=1e-4,
        maxiter=100,
        soft_maxiter=None,
        callback=None,
        minimum_steplen=0.0001,
        maximum_steplen=1.0,
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
        initial_constraint_intensity=None,
        step_constraint_intensity=1.5,
        max_constraint_intensity=1e6,
        initial_constraint_sharpness=None,
        step_constraint_sharpness=1.5,
        max_constraint_sharpness=1e6,
        jumpstart=0,
        jumpstart_split=5,
):
    """
    Makes a series of steps using the BHHH algorithm.

    Parameters
    ----------
    steplen: float
    logger: logging.Logger

    Returns
    -------
    loglike, convergence_tolerance, n_iters, steps
    """
    if logger is None:
        class NoLogger:
            debug = lambda *x: None
            info = lambda *x: None
        logger = NoLogger()

    iter = 0
    steps = []

    if initial_constraint_intensity is not None:
        model.constraint_intensity = initial_constraint_intensity
    if initial_constraint_sharpness is not None:
        model.constraint_sharpness = initial_constraint_sharpness

    if jumpstart:
        current_result = model.loglike2_bhhh(
            leave_out=leave_out, keep_only=keep_only, subsample=subsample,
        )
        current_ll = current_result.ll
        logger.debug(f"before jumpstart loglike {current_ll}")

        model.jumpstart_bhhh(
            jumpstart=jumpstart,
            jumpstart_split=jumpstart_split,
            logger=logger,
        )
        iter += jumpstart

    current_pvals = model.pvals.copy()
    current_result = model.loglike2_bhhh(
        leave_out=leave_out, keep_only=keep_only, subsample=subsample,
    )
    current_ll = current_result.ll
    current_dll = current_result.dll
    current_bhhh = current_result.bhhh

    logger.debug(f"initial loglike {current_ll}")

    def find_direction(current_dll, current_bhhh):
        freedoms = (model.pf.holdfast == 0).to_numpy()
        direction = propose_direction(current_bhhh, np.asarray(current_dll), freedoms)
        tolerance = np.dot(direction, current_dll)
        return direction, tolerance

    direction, tolerance = find_direction(current_dll, current_bhhh)

    message = "Optimization terminated for undetermined reason."

    while True:

        # check various break conditions
        current_constraint_violation = model.constraint_violation(on_violation='return', intensity_check=True)
        if current_constraint_violation:
            logger.debug(f"current constraint violation: {current_constraint_violation}")
        if abs(tolerance) <= ctol and not current_constraint_violation:
            message = "Optimization terminated successfully."
            break
        if iter >= maxiter:
            if current_constraint_violation:
                message = f"Optimization terminated after {iter} iterations with {current_constraint_violation}."
            else:
                message = f"Optimization terminated after {iter} iterations."
            break
        if soft_maxiter is not None:
            if iter >= soft_maxiter and not current_constraint_violation:
                message = f"Optimization terminated after {iter} iterations with no constraint violations."
                break

        # no break, make a step
        iter += 1
        if steps:
            steplen = min(2.0*sum(steps[-momentum:]) / len(steps[-momentum:]), maximum_steplen)
        while True:
            model.set_values(current_pvals + direction * steplen)
            proposed = model.loglike2_bhhh(
                leave_out=leave_out, keep_only=keep_only, subsample=subsample,
            )
            proposed_ll = proposed.ll
            if proposed_ll > current_ll:
                break
            logger.debug(f"failed simple step bhhh {steplen}, degraded {current_ll - proposed_ll}")
            steplen *= 0.5
            if steplen < minimum_steplen:
                break
        if proposed_ll <= current_ll:
            logger.debug("no improvement found, reset to prior x")
            model.set_values(current_pvals)
            if not model.constraint_violation(on_violation='return'):
                logger.debug("-- no constraints are violated")
                # no violations, check that penalty is small
                if np.absolute(current_result.penalty * model.n_cases) > 0.01:
                    logger.debug(f"-- penalty is not small ({current_result.penalty * model.n_cases}), increasing sharpness")
                    # penalty is not small, crank up sharpness
                    proposed.penalty = current_result.penalty
                    while (
                            np.absolute(proposed.penalty * model.n_cases) > 0.01
                            and model.constraint_sharpness < max_constraint_sharpness
                    ):
                        model.constraint_sharpness *= step_constraint_sharpness
                        model.constraint_sharpness = min(
                            model.constraint_sharpness, max_constraint_sharpness
                        )
                        logger.debug(f"-- sharpness to ({model.constraint_sharpness})")
                        proposed = model.loglike2_bhhh(
                            leave_out=leave_out, keep_only=keep_only, subsample=subsample,
                        )
                        proposed_ll = proposed.ll
                        logger.debug(f"   penalty is ({proposed.penalty * model.n_cases})")
                    logger.debug(f"-- proposed_ll {proposed_ll}, current_ll is {current_ll}, will continue")
                else:
                    logger.debug(f"-- penalty is small ({current_result.penalty * model.n_cases}), checking for converge")
                    tolerance = model.constraint_converge_tolerance()
                    logger.debug(f"-- tolerance is {tolerance}")
                    if np.absolute(tolerance) < ctol:
                        message = "Constrained optimization terminated successfully."
                        logger.debug(message)
                        break
                    else:
                        logger.debug(f"-- not yet well converged")
            else:
                raise BHHHSimpleStepFailure(f"simple step bhhh failed\ndirection = {str(direction)}")
        else:
            logger.debug(f"simple step bhhh {steplen} gains {proposed_ll - current_ll} to {proposed_ll}")
        steps.append(steplen)

        current_ll, current_dll, current_bhhh = proposed_ll, proposed.dll, proposed.bhhh
        current_pvals = model.pvals.copy()
        if callback is not None:
            callback(current_pvals)

        if model.constraint_intensity and (
                model.constraint_intensity < max_constraint_intensity
                or model.constraint_sharpness < max_constraint_sharpness
        ):
            logger.debug("increasing constraint intensity and sharpness")
            model.constraint_intensity *= step_constraint_intensity
            model.constraint_sharpness *= step_constraint_sharpness
            model.constraint_intensity = min(
                model.constraint_intensity, max_constraint_intensity
            )
            model.constraint_sharpness = min(
                model.constraint_sharpness, max_constraint_sharpness
            )
            # recompute value with updated penalties
            current_result = model.loglike2_bhhh(
                leave_out=leave_out, keep_only=keep_only, subsample=subsample,
            )
            current_ll = current_result.ll
            current_dll = current_result.dll
            current_bhhh = current_result.bhhh
            logger.debug(f"  constrained convergence tolerance {model.constraint_converge_tolerance()}")

        direction, tolerance = find_direction(current_dll, current_bhhh)
        logger.debug(f"  convergence tolerance {tolerance}")
        logger.debug(f"  constraint intensity {model.constraint_intensity}")
        logger.debug(f"  constraint sharpness {model.constraint_sharpness}")

    return current_ll, tolerance, iter, np.asarray(steps), message
