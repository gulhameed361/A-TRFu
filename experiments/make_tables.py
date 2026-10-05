# -*- coding: utf-8 -*-
"""
make_tables.py -- thesis Tables 7.1, 7.2 and 7.3 from results/runs.csv
(written by run_chapter7.py). Tables 7.4 and 7.5 compare against the
derivative-free solvers and are made by dfo/make_profile.py.

    Table 7.1  A-TRFu (configuration 'full') beside the cited c-TRFu and
               c-TRFi baselines (results/reference/baselines_table71.csv):
               (a) iterations, (b) black-box calls
    Table 7.2  iterations and black-box calls of (a) full, (b) no_hv,
               (c) no_restoration, (d) filter
    Table 7.3  restoration strategies R1 (full) and R2 (full_R2): black-box
               evaluations, final infeasibility and objective

"Max_iter" marks a run that used the 200-iteration budget; "Fail" a run that
stopped at an infeasible point. Writes results/tables/table_7_<n>.csv and
prints the tables.

    python experiments/make_tables.py [--runs results/runs.csv]
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'tables')
BASELINES = os.path.join(ROOT, 'results', 'reference', 'baselines_table71.csv')

LABEL = {
    'powell': 'Powell', 'colville': 'Colville', 'alkylation': 'Alkylation',
    'williams_otto': 'Williams–Otto', 'hetero_scale': 'Hetero_scale',
    'noisy_bb': 'Noisy_bb', 'rosenbrock_bb': 'Rosenbrock',
    'six_bb_chain': 'Six_bb_chain', 'infeasible_start': 'Infeas_start',
    'recycle_loop': 'P_recycle_loop', 'multimodal': 'P_multimodal',
    'ill_conditioned': 'Ill_conditioned', 'active_set_churn': 'Act_set_churn',
    'stiff_bb': 'P_stiff_bb', 'highdim_sep_10': 'Hi_dim_sep_10',
    'highdim_sep_20': 'Hi_dim_sep_20', 'highdim_sep_40': 'Hi_dim_sep_40',
    'highdim_coupled_10': 'Hi_dim_cpd_10', 'highdim_coupled_20': 'Hi_dim_cpd_20',
    'HDA': 'HDA',
}
MAXIMISE = {'alkylation', 'williams_otto', 'recycle_loop'}


def outcome(r):
    """'Max_iter' / 'Fail' for an unsuccessful run, else None."""
    if r['status'] == 'Maximum iterations':
        return 'Max_iter'
    if r['feasible'] != '1':
        return 'Fail'
    return None


def its_evals(r):
    o = outcome(r)
    return (o, o) if o else (r['iterations'], r['bb_evals'])


def write(n, header, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'table_7_{n}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print(f'\nTable 7.{n}  ({os.path.relpath(path, ROOT)})')
    for line in [header] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(line, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(ROOT, 'results', 'runs.csv'))
    args = ap.parse_args()

    runs = {(r['config'], r['problem']): r
            for r in csv.DictReader(open(args.runs, encoding='utf-8'))}
    base = {r['problem']: r
            for r in csv.DictReader(open(BASELINES, encoding='utf-8'))}

    rows = []
    for p, lab in LABEL.items():
        b = base[p]
        rows.append([lab, *its_evals(runs[('full', p)]),
                     b['cTRFu_iterations'], b['cTRFu_bb_calls'],
                     b['cTRFi_iterations'], b['cTRFi_bb_calls']])
    write(1, ['Problem', 'A-TRFu (a)', 'A-TRFu (b)', 'c-TRFu (a)',
              'c-TRFu (b)', 'c-TRFi (a)', 'c-TRFi (b)'], rows)

    configs = ('full', 'no_hv', 'no_restoration', 'filter')
    rows = []
    for p, lab in LABEL.items():
        pairs = [its_evals(runs[(c, p)]) for c in configs]
        rows.append([lab] + [i for i, _ in pairs] + [e for _, e in pairs])
    write(2, ['Problem'] + [f'Iterations {k}' for k in '(a) (b) (c) (d)'.split()]
          + [f'Black-box calls {k}' for k in '(a) (b) (c) (d)'.split()], rows)

    rows = []
    for p, lab in LABEL.items():
        line = [lab, 'max' if p in MAXIMISE else 'min']
        for c in ('full', 'full_R2'):
            r = runs[(c, p)]
            line += [r['bb_evals'], f"{float(r['final_theta']):.2E}",
                     f"{float(r['final_obj']):.7g}"]
        rows.append(line)
    write(3, ['Problem', 'Sense', 'R1 evaluations', 'R1 final theta',
              'R1 objective', 'R2 evaluations', 'R2 final theta',
              'R2 objective'], rows)


if __name__ == '__main__':
    main()
