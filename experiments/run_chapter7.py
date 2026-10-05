# -*- coding: utf-8 -*-
"""
run_chapter7.py -- every A-TRFu run behind thesis Chapter 7 (Tables 7.1-7.5,
Figures 7.3 and 7.5): 20 problems (the 19 benchmark problems + HDA) under five
solver configurations, with one budget of 200 iterations.

    configuration    strategy  hypervolume  restoration    used for
    full             funnel    on           on, R1         Tables 7.1, 7.2(a), 7.3 (R1), 7.4, 7.5
    full_R2          funnel    on           on, R2         Table 7.3 (R2)
    no_hv            funnel    off          on, R1         Table 7.2(b), Figure 7.5
    no_restoration   funnel    on           off            Table 7.2(c)
    filter           filter    on           on, R1         Table 7.2(d)

Each run is executed in its own Python process (as the thesis runs were) and
leaves results/runs/<configuration>/<problem>.json (the solver's result,
including the iteration trajectory) and .log (the full solver output). A run
whose .json already exists is skipped, so an interrupted sweep can simply be
restarted. results/runs.csv collects every finished run.

Usage (from the repository root):
    python experiments/run_chapter7.py                       # everything
    python experiments/run_chapter7.py --configs full        # one configuration
    python experiments/run_chapter7.py --problems powell HDA # some problems
"""
import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('atrfu', 'problems', os.path.join('problems', 'hda')):
    sys.path.insert(0, os.path.join(ROOT, _p))

RUNS_DIR = os.path.join(ROOT, 'results', 'runs')
CSV_PATH = os.path.join(ROOT, 'results', 'runs.csv')

MAX_IT = 200

CONFIGS = {
    'full':           dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_strategy=1),
    'full_R2':        dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_strategy=2),
    'no_hv':          dict(globalization_strategy=1, multiple_black_boxes=0,
                           restoration_strategy=1),
    'no_restoration': dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_strategy=1, restoration_enabled=0),
    'filter':         dict(globalization_strategy=0, multiple_black_boxes=1,
                           restoration_strategy=1),
}

PROBLEM_NAMES = [
    'powell', 'colville', 'alkylation', 'williams_otto', 'hetero_scale',
    'noisy_bb', 'rosenbrock_bb', 'six_bb_chain', 'infeasible_start',
    'recycle_loop', 'multimodal', 'ill_conditioned', 'active_set_churn',
    'stiff_bb', 'highdim_sep_10', 'highdim_sep_20', 'highdim_sep_40',
    'highdim_coupled_10', 'highdim_coupled_20', 'HDA',
]

FIELDS = ['config', 'problem', 'strategy', 'hypervolume', 'restoration',
          'restoration_enabled', 'status', 'feasible', 'iterations',
          'bb_evals', 'bb_evals_detail', 'sim_calls', 'f_steps',
          'theta_steps', 'rejected', 'restorations', 'frp_solves',
          'restoration_phases', 'final_obj', 'final_theta', 'wall_s']


def build(problem):
    """(model, efmap, problem-specific solver kwargs)."""
    if problem == 'HDA':
        from hda_model import build_hda
        return build_hda()
    from benchmark_problems import PROBLEMS
    return PROBLEMS[problem][0]()


def run_one(config, problem):
    """Solve one (configuration, problem) pair in THIS process and write its
    .json and .log."""
    from TRF import TrustRegionSolver

    out_dir = os.path.join(RUNS_DIR, config)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, problem)

    with open(stem + '.log', 'w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(log):
        model, efmap, kwargs = build(problem)
        kwargs.setdefault('max_it', MAX_IT)
        solver = TrustRegionSolver(solver='ipopt', **CONFIGS[config], **kwargs)
        t0 = time.time()
        result = solver.solve(model, efmap)
        result['wall_s'] = time.time() - t0

    if problem == 'HDA':
        # Simulation calls per unit: distinct points at which each unit model
        # was executed (one call returns all of that unit's outputs). Thesis
        # Section 7.6.2 reports these; bb_evals counts scalar outputs.
        from HDA_BB1_Reactor import TRF_CACHE_R
        from HDA_BB2_Flash import TRF_CACHE_FL
        from HDA_BB3_Distillation import TRF_CACHE_DIST
        result['sim_calls'] = {'reactor': len(TRF_CACHE_R),
                               'flash': len(TRF_CACHE_FL),
                               'distillation': len(TRF_CACHE_DIST)}
    result.pop('duals', None)
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)


def _child_env():
    """Environment for a child run: the interpreter's own runtime directories
    on PATH, so IPOPT, gjh and the SciPy DLLs resolve even when the conda
    environment was not activated (Windows otherwise fails silently)."""
    env = dict(os.environ)
    base = os.path.dirname(os.path.abspath(sys.executable))
    extra = [base] + [os.path.join(base, *p) for p in
                      (('Library', 'bin'), ('Library', 'usr', 'bin'),
                       ('Library', 'mingw-w64', 'bin'), ('Scripts',))]
    env['PATH'] = os.pathsep.join([p for p in extra if os.path.isdir(p)]
                                  + [env.get('PATH', '')])
    return env


def row(config, problem):
    path = os.path.join(RUNS_DIR, config, problem + '.json')
    if not os.path.exists(path):
        return None
    r = json.load(open(path, encoding='utf-8'))
    cfg = CONFIGS[config]
    return {
        'config': config,
        'problem': problem,
        'strategy': cfg['globalization_strategy'],
        'hypervolume': cfg['multiple_black_boxes'],
        'restoration': cfg['restoration_strategy'],
        'restoration_enabled': cfg.get('restoration_enabled', 1),
        'status': r['status'],
        'feasible': int(r['feasible']),
        'iterations': r['iterations'],
        'bb_evals': r['bb_evals'],
        'bb_evals_detail': '|'.join(f'{k}:{v}' for k, v in
                                    r['bb_evals_detail'].items()),
        'sim_calls': '|'.join(f'{k}:{v}' for k, v in
                              r.get('sim_calls', {}).items()),
        'f_steps': r['f_steps'],
        'theta_steps': r['theta_steps'],
        'rejected': r['rejected'],
        'restorations': r['restorations'],
        'frp_solves': r['frp_solves'],
        'restoration_phases': r['restoration_phases'],
        'final_obj': repr(r['final_obj']),
        'final_theta': repr(r['final_theta']),
        'wall_s': f"{r['wall_s']:.1f}",
    }


def write_csv():
    rows = [r for c in CONFIGS for p in PROBLEM_NAMES
            for r in [row(c, p)] if r is not None]
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument('--problems', nargs='+', default=PROBLEM_NAMES,
                    choices=PROBLEM_NAMES)
    ap.add_argument('--one', nargs=2, metavar=('CONFIG', 'PROBLEM'),
                    help=argparse.SUPPRESS)      # used for the child process
    args = ap.parse_args()

    if args.one:
        run_one(*args.one)
        return

    for config in args.configs:
        for problem in args.problems:
            if row(config, problem) is not None:
                continue
            print(f'{config:15s} {problem:20s} ...', end=' ', flush=True)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 '--one', config, problem],
                cwd=ROOT, env=_child_env(), capture_output=True, text=True)
            r = row(config, problem)
            if proc.returncode != 0 or r is None:
                tail = (proc.stderr or '').strip().splitlines()[-1:] or ['']
                print(f'FAILED ({tail[0]})', flush=True)
                continue
            print(f"{r['status']} | it={r['iterations']} | "
                  f"evals={r['bb_evals']} | obj={float(r['final_obj']):.6g} | "
                  f"theta={float(r['final_theta']):.2e}", flush=True)
    n = write_csv()
    print(f'\n{n} runs in {CSV_PATH}')


if __name__ == '__main__':
    main()
