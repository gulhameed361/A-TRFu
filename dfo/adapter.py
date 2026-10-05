# -*- coding: utf-8 -*-
"""
adapter.py

Generic Pyomo grey-box -> black-box adapter.

Wraps a (model, efmap) pair built by one of the TRF Code New benchmark
builders (problems/benchmark_problems.py or problems/hda/hda_model.py) so that any of the
six black-box solvers (COBYLA, COBYQA, Nelder-Mead, NEWUOA, Py-BOBYQA, NOMAD)
can optimize it in "full space": the solver controls every Pyomo Var (true
decision variables AND the auxiliary black-box output variables), and every
Constraint (including the y == bb(x) linking equalities) is enforced either
natively (COBYLA/COBYQA/NOMAD) or via a quadratic penalty folded into the
objective (Nelder-Mead/Py-BOBYQA/NEWUOA).
"""
import time
import numpy as np
from pyomo.environ import value, Var, Constraint, minimize

RHO_DEFAULT = 1e6
FEASIBILITY_TOL = 1e-6
LARGE_BOUND = 1e10


class BudgetExhausted(Exception):
    """Raised once the shared black-box evaluation counter hits its cap."""
    pass


class ProblemAdapter:
    def __init__(self, model, efmap, budget=10000, rho=RHO_DEFAULT):
        self.model = model
        self.efmap = efmap
        self.budget = budget
        self.rho = rho

        self.var_list = list(model.component_data_objects(Var, active=True))
        self.n = len(self.var_list)
        self.var_names = [v.name for v in self.var_list]

        self.con_list = list(model.component_data_objects(Constraint, active=True))
        self.n_con = len(self.con_list)

        self.sense_sign = 1.0 if model.obj.sense == minimize else -1.0

        self.x0 = self._get_x0()
        self.lb, self.ub = self._get_bounds()
        self.lb_finite, self.ub_finite = self._finite_bounds(self.lb, self.ub)
        # A few problem builders (e.g. HDA's presolve consistency block) set
        # a Var's initial .value slightly outside its own declared bounds.
        # Every solver here is box-constrained, so clip to a feasible start;
        # PyNomad's C++ core has been observed to segfault on an out-of-bounds
        # x0 rather than reject it gracefully.
        self.x0 = np.clip(self.x0, self.lb_finite, self.ub_finite)

        self.con_kinds = self._classify_constraints()
        self.n_eq = sum(1 for k in self.con_kinds if k[0] == 'eq')
        self.n_ineq = self.n_con - self.n_eq

        self.n_evals = 0
        self._cache_key = None
        self._cache_val = None

        self.best = {
            'penalized': np.inf,
            'raw_obj': None,
            'max_violation': None,
            'x': self.x0.copy(),
        }
        self._start_time = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _get_x0(self):
        x0 = np.empty(self.n)
        for i, v in enumerate(self.var_list):
            val = v.value
            if val is None:
                lb, ub = v.lb, v.ub
                if lb is not None and ub is not None:
                    val = 0.5 * (lb + ub)
                elif lb is not None:
                    val = lb
                elif ub is not None:
                    val = ub
                else:
                    val = 0.0
            x0[i] = float(val)
        return x0

    def _get_bounds(self):
        lb = np.array([v.lb if v.lb is not None else -np.inf for v in self.var_list])
        ub = np.array([v.ub if v.ub is not None else np.inf for v in self.var_list])
        return lb, ub

    @staticmethod
    def _finite_bounds(lb, ub):
        lb_f = np.where(np.isfinite(lb), lb, -LARGE_BOUND)
        ub_f = np.where(np.isfinite(ub), ub, LARGE_BOUND)
        # Some problems (e.g. HDA) have zero-width bounds on effectively-fixed
        # auxiliary variables (lb == ub). Py-BOBYQA requires every bound gap
        # to be >= 2*rhobeg > 0, which a zero-width gap can never satisfy
        # regardless of rhobeg -- nudge such degenerate gaps open by a tiny
        # margin so those solvers can still run (negligible for every other
        # solver, which doesn't require a minimum gap).
        degenerate = (ub_f - lb_f) < 1e-6
        ub_f = np.where(degenerate, lb_f + 1e-6, ub_f)
        return lb_f, ub_f

    def _classify_constraints(self):
        """For each constraint, return (kind, r) where kind in {'eq','le','ge'}
        and r is the RHS bound needed to compute the residual at eval time.
        The 20-problem set has no two-sided (ranged) constraints."""
        kinds = []
        for c in self.con_list:
            if c.equality:
                kinds.append(('eq', c.lower))
            elif c.upper is not None and c.lower is None:
                kinds.append(('le', c.upper))
            elif c.lower is not None and c.upper is None:
                kinds.append(('ge', c.lower))
            else:
                raise ValueError(f"Constraint {c.name} is ranged or unbounded, "
                                  "which this adapter does not support")
        return kinds

    # ------------------------------------------------------------------
    # Core evaluation (single choke point -> counts + caches BB evaluations)
    # ------------------------------------------------------------------
    def set_vector(self, v):
        for var, val in zip(self.var_list, v):
            var.value = float(val)

    def get_vector(self):
        return np.array([value(v) for v in self.var_list])

    def evaluate_point(self, v):
        """Return dict(raw_obj, residuals[list], violations[list],
        max_violation). Caches on the last-seen point so objective+constraint
        queries at the same x count as a single BB evaluation. Raises
        BudgetExhausted once the shared counter reaches self.budget."""
        key = v.tobytes()
        if key == self._cache_key:
            return self._cache_val

        if self.n_evals >= self.budget:
            raise BudgetExhausted(f"Budget of {self.budget} evaluations exhausted")

        self.set_vector(v)
        raw_obj = float(value(self.model.obj.expr))

        residuals = []
        violations = []
        for c, (kind, r) in zip(self.con_list, self.con_kinds):
            body = float(value(c.body))
            if kind == 'eq':
                resid = body - r
                viol = abs(resid)
            elif kind == 'le':
                resid = body - r
                viol = max(resid, 0.0)
            elif kind == 'ge':
                resid = r - body
                viol = max(resid, 0.0)
            residuals.append(resid)
            violations.append(viol)

        max_violation = max(violations) if violations else 0.0

        result = dict(raw_obj=raw_obj, residuals=residuals,
                      violations=violations, max_violation=max_violation)

        self.n_evals += 1
        self._cache_key = key
        self._cache_val = result

        penalized = self._penalized_from(raw_obj, residuals, violations)
        if penalized < self.best['penalized']:
            self.best = dict(penalized=penalized, raw_obj=raw_obj,
                              max_violation=max_violation, x=v.copy())


        return result

    def _penalized_from(self, raw_obj, residuals, violations):
        eq_sq = sum(r**2 for r, (k, _) in zip(residuals, self.con_kinds) if k == 'eq')
        ineq_sq = sum(v**2 for v, (k, _) in zip(violations, self.con_kinds) if k != 'eq')
        return self.sense_sign * raw_obj + self.rho * (eq_sq + ineq_sq)

    # ------------------------------------------------------------------
    # Callables for the solvers
    # ------------------------------------------------------------------
    def penalized_objective(self, v):
        r = self.evaluate_point(np.asarray(v, dtype=float))
        return self._penalized_from(r['raw_obj'], r['residuals'], r['violations'])

    def native_objective(self, v):
        r = self.evaluate_point(np.asarray(v, dtype=float))
        return self.sense_sign * r['raw_obj']

    def native_constraints(self, expand_equalities=False):
        """Return (con_fn, cl, cu) for modopt ProblemLite (COBYQA natively
        supports equalities; COBYLA does not, so pass expand_equalities=True
        to split each equality h(x)=0 into two inequalities h<=0, -h<=0, as
        the reference COBYLA script does). Residuals are defined so that
        `residual <= 0` means feasible for both 'le' and 'ge' kinds."""
        if not expand_equalities:
            cl = np.empty(self.n_con)
            cu = np.empty(self.n_con)
            for i, (kind, _) in enumerate(self.con_kinds):
                if kind == 'eq':
                    cl[i], cu[i] = 0.0, 0.0
                else:  # 'le' or 'ge': residual already signed <= 0 feasible
                    cl[i], cu[i] = -np.inf, 0.0

            def con_fn(v):
                r = self.evaluate_point(np.asarray(v, dtype=float))
                return np.array(r['residuals'])

            return con_fn, cl, cu

        cl_list, cu_list = [], []
        for kind, _ in self.con_kinds:
            if kind == 'eq':
                cl_list += [-np.inf, -np.inf]
                cu_list += [0.0, 0.0]
            else:
                cl_list.append(-np.inf)
                cu_list.append(0.0)
        cl, cu = np.array(cl_list), np.array(cu_list)

        def con_fn(v):
            r = self.evaluate_point(np.asarray(v, dtype=float))
            out = []
            for resid, (kind, _) in zip(r['residuals'], self.con_kinds):
                if kind == 'eq':
                    out.append(resid)
                    out.append(-resid)
                else:
                    out.append(resid)
            return np.array(out)

        return con_fn, cl, cu

    def safe_rhobeg(self, cap=1.0, margin=0.4):
        """A rhobeg guaranteed to satisfy `gap >= 2*rhobeg` for every bounded
        variable (Py-BOBYQA/pdfo requirement), scaled down from `cap`."""
        widths = self.ub_finite - self.lb_finite
        widths = widths[np.isfinite(widths) & (widths > 0)]
        if widths.size == 0:
            return cap
        return float(min(cap, margin * widths.min()))

    def nomad_bb_output_type(self):
        """'OBJ' + one 'PB' per residual entry (equalities contribute two PB
        entries: g and -g)."""
        types = ['OBJ']
        for kind, _ in self.con_kinds:
            if kind == 'eq':
                types.extend(['PB', 'PB'])
            else:
                types.append('PB')
        return types

    def nomad_bb_point(self, x):
        """PyNomad callback: sets BBO string 'f g1 g2 ...' matching
        nomad_bb_output_type(). Returns True/False (False = evaluation
        failure, e.g. budget exhausted or a non-finite black-box output).

        Non-finite (NaN/Inf) outputs are rejected rather than passed through:
        some black boxes here (e.g. HDA's ODE-based reactor model) can
        produce them for extreme/infeasible points, and PyNomad's BBO string
        parser has been observed to crash (native segfault) when handed
        literal 'nan'/'inf' tokens rather than being told the evaluation
        failed."""
        try:
            n = self.n
            v = np.array([x.get_coord(i) for i in range(n)], dtype=float)
            r = self.evaluate_point(v)
            values = [self.sense_sign * r['raw_obj']]
            for resid, (kind, _) in zip(r['residuals'], self.con_kinds):
                if kind == 'eq':
                    values.append(resid)
                    values.append(-resid)
                else:
                    values.append(resid)
            if not all(np.isfinite(val) for val in values):
                return False
            x.setBBO(" ".join(repr(val) for val in values).encode("UTF-8"))
            return True
        except BudgetExhausted:
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    def start_timer(self):
        self._start_time = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self._start_time if self._start_time else None

    def feasible(self, max_violation=None):
        mv = self.best['max_violation'] if max_violation is None else max_violation
        return mv is not None and mv <= FEASIBILITY_TOL
