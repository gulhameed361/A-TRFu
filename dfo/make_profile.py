# -*- coding: utf-8 -*-
"""
make_profile.py -- thesis Table 7.4, Table 7.5 and Figure 7.3 (data profile):
A-TRFu (configuration 'full' of results/runs.csv) against the six
derivative-free solvers (results/dfo_results.csv).

Rules:
  feasible  DFO: maximum constraint violation <= 1e-6;
            A-TRFu: infeasibility theta < 1e-5 (the solver's own test).
  solved    feasible AND objective within 1% of A-TRFu's, measured as
            (f - f*)/(1 + |f*|) (sense-aware; better than A-TRFu also counts).
            A lone feasible DFO result that beats A-TRFu by more than 5% with
            no other DFO solver feasible on that problem is treated as a
            numerical artefact (e.g. an unbounded penalty term), not solved,
            and marked with a dagger in Table 7.4.
  evals     median and maximum black-box evaluations over the solved problems.

Writes results/tables/table_7_4.csv, table_7_5.csv and
results/figures/figure_7_3.png / .svg.

    python dfo/make_profile.py [--runs results/runs.csv] [--dfo results/dfo_results.csv]
"""
import argparse
import csv
import math
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, 'results', 'tables')
FIGURES = os.path.join(ROOT, 'results', 'figures')

DFO = ['NOMAD', 'COBYQA', 'COBYLA', 'Py-BOBYQA', 'NEWUOA', 'Nelder-Mead']
PROFILE_ORDER = ['COBYLA', 'COBYQA', 'Nelder-Mead', 'NEWUOA', 'Py-BOBYQA',
                 'NOMAD', 'A-TRFu']
DISPLAY = {'Nelder-Mead': 'Nelder–Mead'}   # legend name as written in the thesis (en dash)
TOL = 0.01
SPURIOUS_GAP = 0.05
BUDGET = 10000

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


def gap(f, fstar, sense):
    """Relative gap of f against A-TRFu's fstar; negative = better."""
    d = f - fstar if sense == 'minimize' else fstar - f
    return d / (1 + abs(fstar))


def load(runs_csv, dfo_csv):
    atrfu = {r['problem']: r for r in csv.DictReader(open(runs_csv, encoding='utf-8'))
             if r['config'] == 'full'}
    dfo = {(r['Problem'], r['Solver']): r
           for r in csv.DictReader(open(dfo_csv, encoding='utf-8'))}
    out = {}
    for p in LABEL:
        a = atrfu[p]
        sense = dfo[(p, DFO[0])]['ObjSense']
        rec = {'sense': sense,
               'A-TRFu': dict(obj=float(a['final_obj']), evals=int(a['bb_evals']),
                              feasible=a['feasible'] == '1')}
        rec['A-TRFu']['solved'] = rec['A-TRFu']['feasible']
        for s in DFO:
            r = dfo[(p, s)]
            feas = r['Feasible'] == 'True'
            obj = float(r['RawObjective']) if r['RawObjective'] else float('nan')
            rec[s] = dict(obj=obj, evals=int(float(r['BB_Evals_Used'])),
                          feasible=feas, gap=gap(obj, rec['A-TRFu']['obj'], sense))
        feasible_dfo = [s for s in DFO if rec[s]['feasible']]
        for s in DFO:
            d = rec[s]
            d['spurious'] = (d['feasible'] and feasible_dfo == [s]
                             and d['gap'] < -SPURIOUS_GAP)
            d['solved'] = d['feasible'] and not d['spurious'] and d['gap'] <= TOL
        out[p] = rec
    return out


def write(name, header, rows):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print(f'\n{os.path.relpath(path, ROOT)}')
    for line in [header] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(line, widths)))


def table_74(data):
    rows = []
    for p, lab in LABEL.items():
        line = [lab, f"{data[p]['A-TRFu']['obj']:.4g}"]
        for s in DFO:
            d = data[p][s]
            line.append('Infeas.' if not d['feasible'] else
                        f"{d['obj']:.4g}" + (' †' if d['spurious'] else ''))
        rows.append(line)
    write('table_7_4.csv', ['Problem', 'A-TRFu'] + DFO, rows)


def table_75(data):
    n = len(data)
    rows = []
    for s in ['A-TRFu', 'COBYLA', 'COBYQA', 'NEWUOA', 'Nelder-Mead',
              'Py-BOBYQA', 'NOMAD']:
        feas = sum(data[p][s]['feasible'] for p in data)
        solved = [data[p][s]['evals'] for p in data if data[p][s]['solved']]
        med = str(int(math.floor(st.median(solved) + 0.5))) if solved else '–'
        rows.append([s, f'{feas}/{n}', f'{len(solved)}/{n}', med,
                     str(max(solved)) if solved else '–'])
    write('table_7_5.csv', ['Solver', 'Feasible',
                            'Solved (within 1% of A-TRFu)', 'Median evals',
                            'Max evals'], rows)


def figure_73(data):
    """Fraction of problems solved against the evaluation budget."""
    colours = dict(zip(PROFILE_ORDER,
                       plt.cm.tab10(np.linspace(0, 1, len(PROFILE_ORDER)))))
    n = len(data)
    xs = np.logspace(1, np.log10(BUDGET), 500)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in PROFILE_ORDER:
        ev = np.sort([data[p][s]['evals'] for p in data if data[p][s]['solved']])
        ys = [np.sum(ev <= x) / n for x in xs]
        atrfu = s == 'A-TRFu'
        ax.step(xs, ys, where='post', label=DISPLAY.get(s, s), color=colours[s],
                linestyle='-' if atrfu else '--', linewidth=3.0 if atrfu else 1.8)
    ax.set_xscale('log')
    ax.set_xlabel('Black-box evaluation budget', fontsize=13)
    ax.set_ylabel('Fraction of problems solved', fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, which='both')
    ax.tick_params(labelsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=11,
              frameon=True, borderaxespad=0.0)
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ('png', 'svg'):
        path = os.path.join(FIGURES, f'figure_7_3.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', os.path.relpath(path, ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(ROOT, 'results', 'runs.csv'))
    ap.add_argument('--dfo', default=os.path.join(ROOT, 'results', 'dfo_results.csv'))
    args = ap.parse_args()
    data = load(args.runs, args.dfo)
    table_74(data)
    table_75(data)
    figure_73(data)


if __name__ == '__main__':
    main()
