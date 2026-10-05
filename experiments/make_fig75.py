# -*- coding: utf-8 -*-
"""
make_fig75.py -- thesis Figure 7.5: A-TRFu on the HDA flowsheet (a) with and
(b) without hypervolume-based sampling-region management. Per iteration:
infeasibility theta_k, sampling radius sigma_k and step norm (left, log axis)
and the objective (right axis).

Reads <runs>/full/HDA.json and <runs>/no_hv/HDA.json (written by
run_chapter7.py into results/runs/) and writes results/figures/figure_7_5.png
and .svg.

    python experiments/make_fig75.py [--runs results/reference/runs]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'runs')
OUT = os.path.join(ROOT, 'results', 'figures')

PANELS = (('(a)', 'full'), ('(b)', 'no_hv'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=RUNS)
    args = ap.parse_args()
    traj = {}
    for _, config in PANELS:
        with open(os.path.join(args.runs, config, 'HDA.json'), encoding='utf-8') as f:
            traj[config] = json.load(f)['trajectory']
    x_max = max(max(t['iter']) for t in traj.values())

    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    handles = []
    for ax, (label, config) in zip(axes, PANELS):
        t = traj[config]
        it = t['iter']
        h1, = ax.plot(it, t['theta'], '-o', color='blue', ms=4,
                      label='Infeasibility')
        h2, = ax.plot(it, t['sample_radius'], '--s', color='green', ms=4,
                      label='Sampling Region')
        h3, = ax.plot(it, t['step_norm'], '-.^', color='red', ms=4,
                      label='Step Norm')
        ax.set_yscale('log')
        ax.set_xlim(0, x_max)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Infeasibility / Sampling Region / Step Norm')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(label, fontweight='bold')
        ax2 = ax.twinx()
        h4, = ax2.plot(it, t['obj'], ':', color='black', lw=2.5,
                       label='Objective')
        ax2.set_ylabel('Objective Value')
        handles = [h1, h2, h3, h4]
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center',
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'svg'):
        path = os.path.join(OUT, f'figure_7_5.{ext}')
        fig.savefig(path, dpi=300)
        print('wrote', path)


if __name__ == '__main__':
    main()
