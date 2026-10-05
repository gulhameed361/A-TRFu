# -*- coding: utf-8 -*-
"""
run_dfo.py -- the derivative-free comparison of thesis Section 7.6.1: the 20
problems solved by six black-box solvers (COBYLA, COBYQA, Nelder-Mead, NEWUOA,
Py-BOBYQA, NOMAD) with a shared budget of 10,000 evaluations each. The solvers
see the whole model as a black box (full space, every variable a decision
variable); see adapter.py.

Writes results/dfo_results.csv, one row per (problem, solver), after every
run. Pairs already in that file are skipped, so an interrupted sweep can be
restarted, and a subset can be rerun by deleting its rows.

Usage (from the repository root, environment_dfo.yml):
    python dfo/run_dfo.py
    python dfo/run_dfo.py --problems powell HDA --solvers NOMAD
"""
import argparse
import os
import sys
import time
import traceback

import pandas as pd
from pyomo.environ import Var, Constraint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('dfo', 'problems', os.path.join('problems', 'hda')):
    sys.path.insert(0, os.path.join(ROOT, _p))

from benchmark_problems import PROBLEMS   # noqa: E402
from hda_model import build_hda           # noqa: E402
from solvers import SOLVERS               # noqa: E402

BUDGET = 10000
CSV_PATH = os.path.join(ROOT, 'results', 'dfo_results.csv')
COLUMNS = ['Problem', 'Solver', 'n_vars', 'n_eq', 'n_ineq', 'ObjSense',
           'ConstraintHandling', 'RawObjective', 'MaxViolation', 'Feasible',
           'PenalizedObjective', 'BB_Evals_Used', 'Solver_nfev', 'Success',
           'Runtime_s', 'Message']


def registry():
    reg = {name: builder for name, (builder, _desc) in PROBLEMS.items()}
    reg['HDA'] = build_hda
    return reg


def run_pair(builder, runner):
    model, efmap, _kwargs = builder()
    n_vars = len(list(model.component_data_objects(Var, active=True)))
    cons = list(model.component_data_objects(Constraint, active=True))
    n_eq = sum(1 for c in cons if c.equality)
    res = runner(model, efmap, budget=BUDGET)
    return dict(
        n_vars=n_vars, n_eq=n_eq, n_ineq=len(cons) - n_eq,
        ObjSense='minimize' if model.obj.sense == 1 else 'maximize',
        ConstraintHandling=res['constraint_handling'],
        RawObjective=res['raw_obj'], MaxViolation=res['max_violation'],
        Feasible=res['feasible'], PenalizedObjective=res['penalized_obj'],
        BB_Evals_Used=res['bb_evals_used'], Solver_nfev=res['solver_nfev'],
        Success=res['success'], Runtime_s=res['runtime_s'],
        Message=res['message'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problems', nargs='+', default=None)
    ap.add_argument('--solvers', nargs='+', default=list(SOLVERS),
                    choices=list(SOLVERS))
    args = ap.parse_args()

    reg = registry()
    problems = args.problems or list(reg)
    rows = (pd.read_csv(CSV_PATH).to_dict('records')
            if os.path.exists(CSV_PATH) else [])
    done = {(r['Problem'], r['Solver']) for r in rows}

    for problem in problems:
        for solver in args.solvers:
            if (problem, solver) in done:
                continue
            print(f'{problem:20s} {solver:12s} ...', end=' ', flush=True)
            t0 = time.perf_counter()
            row = dict(Problem=problem, Solver=solver)
            try:
                row.update(run_pair(reg[problem], SOLVERS[solver]))
                print(f"feasible={row['Feasible']} obj={row['RawObjective']} "
                      f"evals={row['BB_Evals_Used']}", flush=True)
            except Exception as e:
                row.update(Feasible=False, Success=False,
                           Runtime_s=time.perf_counter() - t0,
                           Message=f'ERROR: {e}\n{traceback.format_exc(limit=3)}')
                print(f'ERROR: {e}', flush=True)
            rows.append(row)
            os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
            pd.DataFrame(rows, columns=COLUMNS).to_csv(CSV_PATH, index=False)


if __name__ == '__main__':
    main()
