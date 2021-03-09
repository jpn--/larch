import numpy as np
import pandas as pd
from .abstract_model import AbstractChoiceModel
from .controller import Model5c
from ..exceptions import MissingDataError, BHHHSimpleStepFailure

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name)


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
        jumpstart=0,
        jumpstart_split=5,
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
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
    try:
        from ..util.timesize import Timer
        from scipy.optimize import minimize
        from .. import _doctest_mode_
        from ..util.rate_limiter import NonBlockingRateLimiter
        from ..util.display import display_head, display_p, display_nothing

        if isinstance(model, Model5c) and model.dataframes is None:
            raise ValueError("you must load data first -- try Model.load_data()")

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
        if isinstance(screen_update_throttle, NonBlockingRateLimiter):
            throttle_gate = screen_update_throttle
        else:
            throttle_gate = NonBlockingRateLimiter(screen_update_throttle)

        if throttle_gate and not quiet and not _doctest_mode_:
            if reuse_tags is None:
                tag1 = display_head(f'Iteration 000 {iteration_number_tail}', level=3)
                tag2 = display_p(f'LL = ...')
                tag3 = display_p('...')
            else:
                tag1, tag2, tag3 = reuse_tags
        else:
            tag1 = display_nothing()
            tag2 = display_nothing()
            tag3 = display_nothing()

        def callback(x, status=None):
            nonlocal iteration_number, throttle_gate
            iteration_number += 1
            if throttle_gate:
                # clear_output(wait=True)
                tag1.update(f'Iteration {iteration_number:03} {iteration_number_tail}')
                tag2.update(f'Best LL = {model._cached_loglike_best}')
                tag3.update(model.pf)
            return False

        if quiet or _doctest_mode_:
            callback = None

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

        if method.lower( )=='bhhh':
            try:
                max_iter = options.get('maxiter' ,100)
                stopping_tol = options.get('ctol' ,1e-5)

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
                tag1.update(f'Iteration {iteration_number:03} [BHHH Not Available] {iteration_number_tail}', force=True)
                tag3.update(model.pf, force=True)
                if method2 is not None:
                    method_used = f"{method2}"
                    method = method2
            except BHHHSimpleStepFailure:
                tag1.update(f'Iteration {iteration_number:03} [Exception Recovery] {iteration_number_tail}', force=True)
                tag3.update(model.pf, force=True)
                if method2 is not None:
                    method_used = f"{method_used}|{method2}"
                    method = method2
            except:
                tag1.update(f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}', force=True)
                tag3.update(model.pf, force=True)
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

                raw_result = minimize(
                    model.neg_loglike,
                    model.pvals,
                    args=(0, -1, 1, leave_out, keep_only, subsample), # start_case, stop_case, step_case, leave_out, keep_only, subsample
                    method=method,
                    jac=model.neg_d_loglike,
                    bounds=bounds,
                    callback=callback,
                    options=options,
                    constraints=constraints,
                    **kwargs
                )
            except:
                tag1.update(f'Iteration {iteration_number:03} [Exception] {iteration_number_tail}', force=True)
                tag3.update(model.pf, force=True)
                raise

        timer.stop()

        if final_screen_update and not quiet and not _doctest_mode_ and raw_result is not None:
            converged = raw_result.get("message", "Converged")
            tag1.update(f'Iteration {iteration_number:03} [{converged}] {iteration_number_tail}', force=True)
            tag2.update(f'Best LL = {model._cached_loglike_best}', force=True)
            tag3.update(model.pf, force=True)

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
                result['d_loglike'] = pd.Series(-v, index=model.pnames)
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

        if return_tags:
            return result, tag1, tag2, tag3

        return result

    except:
        logger.exception("error in maximize_loglike")
        raise
