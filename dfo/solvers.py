# -*- coding: utf-8 -*-
"""
solvers.py

One runner per black-box solver (COBYLA, COBYQA, Nelder-Mead, NEWUOA,
Py-BOBYQA, NOMAD). Each takes a ProblemAdapter and a shared evaluation
budget, wires the solver, and returns a standard result dict:

    success, message, raw_obj, max_violation, feasible, penalized_obj,
    bb_evals_used, solver_nfev, runtime_s, constraint_handling, x_star
"""
import warnings

from adapter import ProblemAdapter, BudgetExhausted

warnings.filterwarnings('ignore')


def _standard_result(adapter, success, message, solver_nfev,
                      constraint_handling, x_star=None):
    best = adapter.best
    return dict(
        success=bool(success),
        message=str(message),
        raw_obj=best['raw_obj'],
        max_violation=best['max_violation'],
        feasible=adapter.feasible(),
        penalized_obj=best['penalized'],
        bb_evals_used=adapter.n_evals,
        solver_nfev=solver_nfev,
        runtime_s=adapter.elapsed(),
        constraint_handling=constraint_handling,
        x_star=(x_star if x_star is not None else best['x']),
    )


def _make_problem_lite(mo, adapter, expand_equalities=False):
    """modopt's ProblemLite rejects an explicitly empty constraint array, so
    omit con/cl/cu entirely for unconstrained problems (e.g. 'powell')."""
    kwargs = dict(x0=adapter.x0, obj=adapter.native_objective,
                  xl=adapter.lb_finite, xu=adapter.ub_finite)
    if adapter.n_con > 0:
        con_fn, cl, cu = adapter.native_constraints(expand_equalities=expand_equalities)
        kwargs.update(con=con_fn, cl=cl, cu=cu)
    return mo.ProblemLite(**kwargs)


def run_cobyla(model, efmap, budget=10000, rho=1e6):
    import modopt as mo
    from modopt import COBYLA

    adapter = ProblemAdapter(model, efmap, budget=budget, rho=rho)
    adapter.start_timer()
    # COBYLA has no native equality support -> split h=0 into h<=0, -h<=0
    problem = _make_problem_lite(mo, adapter, expand_equalities=True)
    optimizer = COBYLA(problem, solver_options={
        'maxiter': budget, 'catol': 1e-6, 'disp': False})

    try:
        optimizer.solve()
        res = optimizer.results
        return _standard_result(adapter, res.get('success', False),
                                 res.get('message', ''), res.get('nfev'),
                                 'native')
    except BudgetExhausted as e:
        return _standard_result(adapter, False, str(e), adapter.n_evals, 'native')
    except Exception as e:
        return _standard_result(adapter, False, f"error: {e}", adapter.n_evals, 'native')


def run_cobyqa(model, efmap, budget=10000, rho=1e6):
    import modopt as mo
    from modopt import COBYQA

    adapter = ProblemAdapter(model, efmap, budget=budget, rho=rho)
    adapter.start_timer()
    problem = _make_problem_lite(mo, adapter)
    optimizer = COBYQA(problem, solver_options={
        'maxiter': budget, 'feasibility_tol': 1e-8, 'disp': False})

    try:
        optimizer.solve()
        res = optimizer.results
        return _standard_result(adapter, res.get('success', False),
                                 res.get('message', ''), res.get('nfev'),
                                 'native')
    except BudgetExhausted as e:
        return _standard_result(adapter, False, str(e), adapter.n_evals, 'native')
    except Exception as e:
        return _standard_result(adapter, False, f"error: {e}", adapter.n_evals, 'native')


def run_nelder_mead(model, efmap, budget=10000, rho=1e6):
    from scipy.optimize import minimize

    adapter = ProblemAdapter(model, efmap, budget=budget, rho=rho)
    adapter.start_timer()
    bounds = list(zip(adapter.lb_finite, adapter.ub_finite))

    try:
        result = minimize(
            adapter.penalized_objective, adapter.x0, method='Nelder-Mead',
            bounds=bounds,
            options={'maxfev': budget, 'maxiter': budget,
                     'disp': False, 'xatol': 1e-8, 'fatol': 1e-8},
        )
        return _standard_result(adapter, result.success, result.message,
                                 result.nfev, 'penalty', x_star=result.x)
    except BudgetExhausted as e:
        return _standard_result(adapter, False, str(e), adapter.n_evals, 'penalty')
    except Exception as e:
        return _standard_result(adapter, False, f"error: {e}", adapter.n_evals, 'penalty')


def run_newuoa(model, efmap, budget=10000, rho=1e6):
    from pdfo import pdfo

    adapter = ProblemAdapter(model, efmap, budget=budget, rho=rho)
    adapter.start_timer()

    try:
        res = pdfo(
            fun=adapter.penalized_objective, x0=adapter.x0, method='newuoa',
            options={'maxfev': budget, 'rhobeg': 1.0, 'rhoend': 1e-6, 'quiet': True},
        )
        return _standard_result(adapter, res.success, res.message,
                                 res.nfev, 'penalty', x_star=res.x)
    except BudgetExhausted as e:
        return _standard_result(adapter, False, str(e), adapter.n_evals, 'penalty')
    except Exception as e:
        return _standard_result(adapter, False, f"error: {e}", adapter.n_evals, 'penalty')


def run_py_bobyqa(model, efmap, budget=10000, rho=1e6):
    import pybobyqa

    adapter = ProblemAdapter(model, efmap, budget=budget, rho=rho)
    adapter.start_timer()
    rhobeg = adapter.safe_rhobeg(cap=1.0)

    try:
        soln = pybobyqa.solve(
            adapter.penalized_objective, adapter.x0,
            bounds=(adapter.lb_finite, adapter.ub_finite),
            rhobeg=rhobeg, rhoend=min(1e-6, rhobeg * 1e-6), maxfun=budget,
            objfun_has_noise=False, seek_global_minimum=False,
            print_progress=False,
        )
        success = (soln.flag == soln.EXIT_SUCCESS)
        return _standard_result(adapter, success, soln.msg, soln.nf,
                                 'penalty', x_star=soln.x)
    except BudgetExhausted as e:
        return _standard_result(adapter, False, str(e), adapter.n_evals, 'penalty')
    except Exception as e:
        return _standard_result(adapter, False, f"error: {e}", adapter.n_evals, 'penalty')


def run_nomad(model, efmap, budget=10000, rho=1e6):
    import PyNomad

    adapter = ProblemAdapter(model, efmap, budget=budget, rho=rho)
    adapter.start_timer()

    bb_output_type = " ".join(adapter.nomad_bb_output_type())
    params = [
        f"DIMENSION {adapter.n}",
        f"BB_OUTPUT_TYPE {bb_output_type}",
        f"MAX_BB_EVAL {budget}",
        "EPSILON 1e-8",
        "DISPLAY_DEGREE 0",
        "MEGA_SEARCH_POLL yes",
    ]
    x0 = list(adapter.x0)
    lb = list(adapter.lb_finite)
    ub = list(adapter.ub_finite)

    try:
        result = PyNomad.optimize(adapter.nomad_bb_point, x0, lb, ub, params)
        message = result.get('run_status', '') if isinstance(result, dict) else ''
        success = adapter.n_evals > 0 and adapter.best['raw_obj'] is not None
        return _standard_result(adapter, success, message, adapter.n_evals,
                                 'native')
    except BudgetExhausted as e:
        return _standard_result(adapter, False, str(e), adapter.n_evals, 'native')
    except Exception as e:
        return _standard_result(adapter, False, f"error: {e}", adapter.n_evals, 'native')


SOLVERS = {
    'COBYLA': run_cobyla,
    'COBYQA': run_cobyqa,
    'Nelder-Mead': run_nelder_mead,
    'NEWUOA': run_newuoa,
    'Py-BOBYQA': run_py_bobyqa,
    'NOMAD': run_nomad,
}
