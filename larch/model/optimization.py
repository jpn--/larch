import numpy as np
import pandas as pd
from .abstract_model import AbstractChoiceModel
from .controller import Model5c
from ..exceptions import MissingDataError, BHHHSimpleStepFailure

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name)


class ModelDashboard:
    """
    Current state of the model.

    This dashboard is displayed on creation.
    """

    def __init__(self, throttle=2, visible=True, *, tags=None):
        from ..util.display import display_head, display_p, display_nothing
        from ..util.rate_limiter import NonBlockingRateLimiter
        self.visible = visible
        if tags is not None:
            self.head, self.subhead, self.body = tags
        else:
            if visible:
                self.head = display_head(f'Larch Model Dashboard', level=3)
                self.subhead = display_p(f'LL = ...')
                self.body = display_p('...')
            else:
                self.head = display_nothing()
                self.subhead = display_nothing()
                self.body = display_nothing()
        if isinstance(throttle, NonBlockingRateLimiter):
            self.throttle_gate = throttle
        else:
            self.throttle_gate = NonBlockingRateLimiter(throttle)

    def update(self, head=None, subhead=None, body=None, force=False):
        if self.visible:
            if (self.throttle_gate or force or 1):
                if head is not None:
                    self.head.update(head, force=force)
                if subhead is not None:
                    self.subhead.update(subhead, force=force)
                if body is not None:
                    self.body.update(body, force=force)


def maximize_loglike(
        model,
        method=None,
        method2=None,
        quiet=False,
        screen_update_throttle=2,
        final_screen_update=True,
        check_for_overspecification=True,
        return_tags=False,
        reuse_tags=None,
        iteration_number=0,
        iteration_number_tail="",
        options=None,
        maxiter=None,
        bhhh_start=0,
        jumpstart=0,
        jumpstart_split=5,
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
        return_dashboard=False,
        dashboard=None,
        prior_result=None,
        **kwargs,
):
    """
    Maximize the log likelihood.


    Parameters
    ----------
    model : AbstractChoiceModel
        The data for this model should previously have been
        prepared using the `load_data` method.
    method : str, optional
        The optimization method to use.  See scipy.optimize for
        most possibilities, or use 'BHHH'. Defaults to SLSQP if
        there are any constraints or finite parameter bounds,
        otherwise defaults to BHHH.
    quiet : bool, default False
        Whether to suppress the dashboard.

    Returns
    -------
    dictx
        A dictionary of results, including final log likelihood,
        elapsed time, and other statistics.  The exact items
        included in output will vary by estimation method.

    Raises
    ------
    ValueError
        If the `dataframes` are not already loaded.

    """
    _initial_constraint_intensity = getattr(model, 'constraint_intensity', None)
    _initial_constraint_sharpness = getattr(model, 'constraint_sharpness', None)
    try:
        from ..util.timesize import Timer
        from scipy.optimize import minimize
        from .. import _doctest_mode_
        from ..util.rate_limiter import NonBlockingRateLimiter
        from ..util.display import display_head, display_p, display_nothing

        if isinstance(model, Model5c) and model.dataframes is None:
            raise ValueError("you must load data first -- try Model.load_data()")

        if prior_result is not None:
            dashboard = dashboard or prior_result.get('dashboard', None)
            iteration_number = iteration_number or prior_result.get('iteration_number', 0)

        if _doctest_mode_:
            from ..model import Model
            if type(model) == Model:
                model.unmangle()
                model._frame.sort_index(inplace=True)
                model.unmangle(True)

        if options is None:
            options = {}
        if maxiter is not None:
            options['maxiter'] = maxiter

        timer = Timer()

        if not quiet and not _doctest_mode_:
            if isinstance(reuse_tags, tuple) and len(reuse_tags) == 3 and dashboard is None:
                dashboard = ModelDashboard(screen_update_throttle, tags=reuse_tags)
            if dashboard is not None:
                if not isinstance(dashboard, ModelDashboard):
                    raise ValueError(f"dashboard must be ModelDashboard, not {type(dashboard)}")
            else:
                dashboard = ModelDashboard(screen_update_throttle)
        else:
            dashboard = ModelDashboard(visible=False)

        def callback(x, status=None):
            nonlocal iteration_number, dashboard, method
            iteration_number += 1
            if isinstance(status, dict) and 'penalty' in status:
                logger.critical(
                    f'Iteration {iteration_number:03} {iteration_number_tail}: '
                    f'Currently using {method}, '
                    f'Total LL = {status["total_loglike"]}, '
                    f'Constraint Penalty = {status["penalty"]}'
                )
                dashboard.update(
                    f'Iteration {iteration_number:03} {iteration_number_tail}',
                    (
                        f'Currently using {method}, '
                        f'Total LL = {status["total_loglike"]}, '
                        f'Constraint Penalty = {status["penalty"]}'
                    ),
                    model.pf,
                )
            else:
                dashboard.update(
                    f'Iteration {iteration_number:03} {iteration_number_tail}',
                    f'Currently using {method}, Best LL = {model._cached_loglike_best}',
                    model.pf,
                )
            return False

        if quiet or _doctest_mode_:
            callback = None

        if bhhh_start:
            if method is None or method.lower() == 'bhhh':
                method = 'slsqp'

        if method is None:
            try:
                has_constraints = bool(model.constraints)
            except AttributeError:
                has_constraints = False
            if has_constraints or np.isfinite(model.pf['minimum'].max()) or np.isfinite(model.pf['maximum'].min()):
                method = 'slsqp'
            else:
                method = 'bhhh'

        if method2 is None and method.lower() == 'bhhh':
            method2 = 'slsqp'

        method_used = method
        raw_result = None

        if bhhh_start:
            try:
                _restore_method = method
                method_used = f"bhhh({bhhh_start})->{method}"
                method = f"bhhh({bhhh_start})"
                current_ll, tolerance, iter, steps_bhhh, message = model.fit_bhhh(
                    steplen=1.0,
                    momentum=5,
                    logger=None,
                    ctol=1e-4,
                    maxiter=options.get('maxiter' ,100),
                    soft_maxiter=bhhh_start,
                    callback=callback,
                    minimum_steplen=0.0001,
                    maximum_steplen=1.0,
                    leave_out=-1,
                    keep_only=-1,
                    subsample=-1,
                    initial_constraint_intensity=1.0,
                    step_constraint_intensity=1.5,
                    max_constraint_intensity=1e6,
                    initial_constraint_sharpness=1.0,
                    step_constraint_sharpness=1.5,
                    max_constraint_sharpness=1e6,
                )
                raw_result = {
                    'loglike': current_ll,
                    'x': model.pvals,
                    'tolerance': tolerance,
                    'steps': steps_bhhh,
                    'message': message,
                }
            finally:
                method = _restore_method
                model.constraint_intensity = 0.0

        if method.lower( )=='bhhh':
            try:
                max_iter = options.get('maxiter' ,100)
                stopping_tol = options.get('ctol' ,1e-5)

                if hasattr(model, 'fit_bhhh'):
                    current_ll, tolerance, iter_bhhh, steps_bhhh, message = model.fit_bhhh(
                        # steplen=1.0,
                        # momentum=5,
                        ctol=stopping_tol,
                        maxiter=max_iter,
                        callback=callback,
                        leave_out=leave_out,
                        keep_only=keep_only,
                        subsample=subsample,
                        # initial_constraint_intensity=1.0,
                        # step_constraint_intensity=1.5,
                        # max_constraint_intensity=1e6,
                        # initial_constraint_sharpness=1.0,
                        # step_constraint_sharpness=1.5,
                        # max_constraint_sharpness=1e6,
                    )
                else:
                    current_ll, tolerance, iter_bhhh, steps_bhhh, message = model.simple_fit_bhhh(
                        ctol=stopping_tol,
                        maxiter=max_iter,
                        callback=callback,
                        jumpstart=jumpstart,
                        jumpstart_split=jumpstart_split,
                        leave_out=leave_out,
                        keep_only=keep_only,
                        subsample=subsample,
                    )
                raw_result = {
                    'loglike' :current_ll,
                    'x': model.pvals,
                    'tolerance' :tolerance,
                    'steps' :steps_bhhh,
                    'message' :message,
                }
            except NotImplementedError:
                dashboard.update(
                    f'Iteration {iteration_number:03} [BHHH Not Available] {iteration_number_tail}',
                    body=model.pf,
                    force=True,
                )
                if method2 is not None:
                    method_used = f"{method2}"
                    method = method2
            except BHHHSimpleStepFailure:
                dashboard.update(
                    f'Iteration {iteration_number:03} [Exception Recovery] {iteration_number_tail}',
                    body=model.pf,
                    force=True,
                )
                if method2 is not None:
                    method_used = f"{method_used}|{method2}"
                    method = method2
            except:
                dashboard.update(
                    f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}',
                    body=model.pf,
                    force=True,
                )
                raise

        if method.lower() != 'bhhh':
            try:
                bounds = None
                if isinstance(method ,str) and method.lower() in ('slsqp', 'l-bfgs-b', 'tnc', 'trust-constr'):
                    bounds = model.pbounds
                    if np.any(np.isinf(model.pf.minimum)) or np.any(np.isinf(model.pf.maximum)):
                        import warnings
                        warnings.warn( # infinite bounds # )
                            f"{method} may not play nicely with unbounded parameters\n"
                            "if you get poor results, consider setting global bounds with model.set_cap()"
                        )

                try:
                    constraints = model._get_constraints(method)
                except:
                    constraints = ()

                args = getattr(model, '_null_slice', (0,-1,1))
                raw_result = minimize(
                    model.neg_loglike,
                    model.pvals,
                    args=args+(leave_out, keep_only, subsample), # start_case, stop_case, step_case, leave_out, keep_only, subsample
                    method=method,
                    jac=model.neg_d_loglike,
                    bounds=bounds,
                    callback=callback,
                    options=options,
                    constraints=constraints,
                    **kwargs
                )
            except:
                dashboard.update(
                    f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}',
                    body=model.pf,
                    force=True,
                )
                raise

        timer.stop()

        if final_screen_update and not quiet and not _doctest_mode_ and raw_result is not None:
            converged = raw_result.get("message", "Converged")
            dashboard.update(
                f'Iteration {iteration_number:03} [{converged}] {iteration_number_tail}',
                f'Best LL = {model._cached_loglike_best}',
                model.pf,
                force=True,
            )

        if raw_result is None:
            raw_result = {}
        # if check_for_overspecification:
        #	model.check_for_possible_overspecification()

        from ..util import dictx
        result = dictx()
        for k ,v in raw_result.items():
            if k == 'fun':
                result['loglike'] = -v
            elif k == 'jac':
                try:
                    result['d_loglike'] = pd.Series(-v, index=model.pnames)
                except TypeError:
                    result[k] = v
            elif k == 'x':
                result['x'] = pd.Series(v, index=model.pnames)
            else:
                result[k] = v
        result['elapsed_time'] = timer.elapsed()
        result['method'] = method_used
        try:
            result['n_cases'] = model.n_cases
        except NotImplementedError:
            pass
        result['iteration_number'] = iteration_number

        if 'loglike' in result:
            result['logloss'] = -result['loglike'] / model.total_weight()

        if _doctest_mode_:
            result['__verbose_repr__'] = True

        model._most_recent_estimation_result = result

        if return_dashboard:
            result['dashboard'] = dashboard

        if return_tags:
            return result, dashboard.head, dashboard.subhead, dashboard.body


        return result

    except:
        logger.exception("error in maximize_loglike")
        raise
    finally:
        if _initial_constraint_intensity is not None:
            setattr(model, 'constraint_intensity', _initial_constraint_intensity)
        if _initial_constraint_sharpness is not None:
            setattr(model, 'constraint_sharpness', _initial_constraint_sharpness)


def propose_direction(bhhh, dloglike, freedoms):
    direction = np.zeros_like(dloglike)
    # try:
    #     direction1 = np.linalg.solve(bhhh[freedoms, :][:, freedoms], dloglike[freedoms])
    # except np.linalg.LinAlgError:
    direction1 = np.linalg.lstsq(bhhh[freedoms, :][:, freedoms], dloglike[freedoms])[0]
    try:
        direction[freedoms] = direction1
    except:
        print("direction", direction.shape)
        print("direction1", direction1.shape)
        print("freedoms", freedoms.shape)
        raise
    return direction



def fit_bhhh(
        model,
        steplen=1.0,
        momentum=5,
        printer=None,
        ctol=1e-4,
        maxiter=100,
        callback=None,
        jumpstart=0,
        jumpstart_split=5,
        minimum_steplen=0.0001,
        maximum_steplen=1.0,
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
        initial_constraint_intensity=None,
        step_constraint_intensity=1.5,
        initial_constraint_sharpness=None,
        step_constraint_sharpness=1.2,
):
    """
    Makes a series of steps using the BHHH algorithm.

    Parameters
    ----------
    steplen: float
    printer: callable

    Returns
    -------
    loglike, convergence_tolerance, n_iters, steps
    """
    current_pvals = model.pvals.copy()
    iter = 0
    steps = []

    if initial_constraint_intensity is not None:
        model.constraint_intensity = initial_constraint_intensity
    if initial_constraint_sharpness is not None:
        model.constraint_sharpness = initial_constraint_sharpness

    # if jumpstart:
    #     model.jumpstart_bhhh(jumpstart=jumpstart, jumpstart_split=jumpstart_split)
    #     iter += jumpstart

    current_ll, current_dll, current_bhhh = model._loglike2_bhhh_tuple(
        leave_out=leave_out, keep_only=keep_only, subsample=subsample,
    )

    def find_direction(current_dll, current_bhhh):
        freedoms = (model.pf.holdfast == 0).to_numpy()
        direction = propose_direction(current_bhhh, np.asarray(current_dll), freedoms)
        tolerance = np.dot(direction, current_dll)
        return direction, tolerance

    direction, tolerance = find_direction(current_dll, current_bhhh)

    while abs(tolerance) > ctol and iter < maxiter:
        iter += 1
        if steps:
            steplen = min(2.0*sum(steps[-momentum:]) / len(steps[-momentum:]), maximum_steplen)
        while True:
            model.set_values(current_pvals + direction * steplen)
            proposed_ll, proposed_dll, proposed_bhhh = model._loglike2_bhhh_tuple(
                leave_out=leave_out, keep_only=keep_only, subsample=subsample,
            )
            if proposed_ll > current_ll: break
            steplen *= 0.5
            if steplen < minimum_steplen: break
        if proposed_ll <= current_ll:
            model.set_values(current_pvals)
            raise BHHHSimpleStepFailure(f"simple step bhhh failed\ndirection = {str(direction)}")
        if printer is not None:
            printer("simple step bhhh {} to gain {}".format(steplen, proposed_ll - current_ll))
        steps.append(steplen)

        current_ll, current_dll, current_bhhh = proposed_ll, proposed_dll, proposed_bhhh
        current_pvals = model.pvals.copy()
        if callback is not None:
            callback(current_pvals)

        model.constraint_intensity *= step_constraint_intensity
        model.constraint_sharpness *= step_constraint_sharpness
        direction, tolerance = find_direction(current_dll, current_bhhh)

    if abs(tolerance) <= ctol:
        message = "Optimization terminated successfully."
    else:
        message = f"Optimization terminated after {iter} iterations."

    return current_ll, tolerance, iter, np.asarray(steps), message
