# A-TRFu: advanced trust-region funnel method for grey-box optimisation

This repository contains the A-TRFu solver (Algorithm 7.1 of the thesis) and
everything needed to reproduce the results of **Chapter 7**: Tables 7.1–7.5 and
Figures 7.3 and 7.5.

A grey-box problem couples equation-oriented ("glass-box") constraints with
"black-box" truth models that can only be evaluated. A-TRFu replaces each black
box by a local linear surrogate built from finite differences inside a sampling
region, solves trust-region subproblems on the surrogate model, and globalises
with a funnel. The funnel limits infeasibility, measured as the mismatch between
surrogate and truth. Two mechanisms are specific to A-TRFu:

- **Hypervolume sampling-region management**: the infeasibility of each black
  box is tracked separately, and the sampling radius is set from the
  hypervolume of their Pareto front (Section 7.2).
- **Feasibility restoration** (7.11): after two consecutive rejected steps, a
  restoration problem is solved. Restoration Strategy R1 (the default) or R2
  sets the trust-region radius during it (Section 7.3).

The filter variant (strategy 0) is kept for the comparison in Table 7.2(d).

## Layout

```
atrfu/                     the solver
  TRF.py                   TrustRegionSolver (configuration) and the main loop
  PyomoInterface.py        model reformulation, surrogates, subproblems, criticality measure
  funnelMethod.py          funnel
  filterMethod.py          filter
  HV_calculations.py       Pareto front and hypervolume
  GJHPseudoSolver.py       derivatives via AMPL gjh (installed automatically, see below)
  readgjh.py, Logger.py, helper.py
problems/
  benchmark_problems.py    the 19 benchmark problems
  hda/                     problem 20: the hydrodealkylation flowsheet (hda_model.py)
                           and its reactor, flash and distillation black boxes
experiments/
  run_chapter7.py          every A-TRFu run of Chapter 7
  make_tables.py           Tables 7.1, 7.2, 7.3
  make_fig75.py            Figure 7.5
dfo/
  run_dfo.py               the six derivative-free solvers on the 20 problems
  adapter.py, solvers.py   problem adapter and solver settings
  make_profile.py          Tables 7.4, 7.5 and Figure 7.3
results/reference/         the reported results (see "Reference results")
environment.yml            solver environment
environment_dfo.yml        derivative-free comparison environment
```

## Requirements

**Solver and Chapter 7 runs** (`environment.yml`):

- Python 3.8 with Pyomo 6.2;
- the IPOPT executable (the reported runs used IPOPT 3.14.13 with MUMPS, built
  with MinGW-w64 under MSYS2). On Linux and macOS `environment.yml` installs it;
  on Windows the conda-forge package provides only the library, so put an
  `ipopt.exe` on the `PATH`, for example from the COIN-OR Ipopt releases on
  GitHub;
- AMPL's `gjh`, which evaluates gradients and Jacobians. If no `gjh` is found
  on the `PATH`, the solver downloads it on first use from AMPL's package index
  (`ampl-module-gjh`) and places it in `atrfu/`. Without internet access, install
  it with `pip install ampl-module-gjh --index-url https://pypi.ampl.com` or
  put a `gjh` executable on the `PATH`.

```
conda env create -f environment.yml
conda activate atrfu
```

**Derivative-free comparison** (`environment_dfo.yml`, Python 3.12): the six
solvers COBYLA and COBYQA (through modOpt), Nelder–Mead (SciPy), NEWUOA (PDFO),
Py-BOBYQA and NOMAD (PyNomadBBO).

```
conda env create -f environment_dfo.yml
```

All commands below are run from the repository root.

## Reproducing Chapter 7

### 1. A-TRFu runs (Tables 7.1–7.3, Figure 7.5)

```
conda activate atrfu
python experiments/run_chapter7.py
python experiments/make_tables.py
python experiments/make_fig75.py
```

`run_chapter7.py` solves the 20 problems under five configurations, with one
budget of 200 iterations:

| Configuration | Globalisation | Hypervolume | Restoration | Used in |
|---|---|---|---|---|
| `full` | funnel | on | R1 | Tables 7.1, 7.2(a), 7.3 (R1), 7.4, 7.5; Figure 7.5(a) |
| `full_R2` | funnel | on | R2 | Table 7.3 (R2) |
| `no_hv` | funnel | off | R1 | Table 7.2(b); Figure 7.5(b) |
| `no_restoration` | funnel | on | off | Table 7.2(c) |
| `filter` | filter | on | R1 | Table 7.2(d) |

Each run executes in its own process. It writes `results/runs/<configuration>/<problem>.json`
(the solver's result, including the per-iteration trajectory) and `.log` (the
full solver output). All finished runs are collected in `results/runs.csv`.
Finished runs are skipped, so an interrupted sweep can be restarted, and
`--configs` and `--problems` select a subset. The 100 runs take about an hour
on a current laptop (47 minutes of solver time; the longest single run is HDA
under R2, about 3 minutes).

`make_tables.py` writes `results/tables/table_7_1.csv` to `table_7_3.csv`, and
`make_fig75.py` writes `results/figures/figure_7_5.png` and `.svg`.

The c-TRFu and c-TRFi columns of Table 7.1 are cited from the earlier
implementation, which is not part of this repository. They are given in
`results/reference/baselines_table71.csv` together with two corrections to the
printed table (see the `note` column).

### 2. Derivative-free comparison (Tables 7.4, 7.5, Figure 7.3)

```
conda activate atrfu-dfo
python dfo/run_dfo.py
python dfo/make_profile.py
```

`run_dfo.py` gives each solver 10,000 black-box evaluations per problem. The
solvers see the whole model as a black box: every variable is a decision
variable, and every constraint, including the black-box links, is enforced
natively (COBYLA, COBYQA, NOMAD) or by a quadratic penalty (the others). Results
go to `results/dfo_results.csv` after every run; pairs already present are
skipped. The full sweep records about three and a half hours of solver time,
half of it NOMAD, mostly on the high-dimensional problems and HDA.

`make_profile.py` reads `results/runs.csv` (configuration `full`) and
`results/dfo_results.csv`. It writes `results/tables/table_7_4.csv`,
`table_7_5.csv` and `results/figures/figure_7_3.png` / `.svg`. A problem counts
as solved by a derivative-free solver when the point is feasible (maximum
violation ≤ 10⁻⁶) and its objective is within 1% of A-TRFu's, measured as
(f − f*)/(1 + |f*|). A lone feasible result that beats A-TRFu by more than 5%,
with no other derivative-free solver feasible on that problem, is treated as a
numerical artefact and marked † in Table 7.4 (Colville, NEWUOA).

To build the tables from the reported results instead of new runs:

```
python experiments/make_tables.py --runs results/reference/runs.csv
python dfo/make_profile.py --runs results/reference/runs.csv --dfo results/reference/dfo_results.csv
```

## Reference results

`results/reference/` holds the results reported in the thesis:

- `runs.csv`: the 100 A-TRFu runs, in the format of `results/runs.csv`;
- `runs/full/HDA.json`, `runs/no_hv/HDA.json`: the two HDA runs of Figure 7.5,
  with their trajectories (`python experiments/make_fig75.py --runs results/reference/runs`);
- `dfo_results.csv`: the 120 derivative-free runs, made with `dfo/run_dfo.py`
  as shipped. The runs are deterministic: `run_dfo.py` reproduces every
  column of this file except the runtimes;
- `baselines_table71.csv`: the cited c-TRFu / c-TRFi values.

The reported A-TRFu runs were made on Windows 11 with Python 3.8.20, Pyomo 6.2,
NumPy 1.24.4 and IPOPT 3.14.13. A different IPOPT build, linear solver or
platform can change iteration counts on the harder problems: the subproblems
are solved to a tolerance, and their solutions steer the iterates. Final
objectives agree far more closely than paths. In the reference environment,
`run_chapter7.py` reproduces every row of `results/reference/runs.csv`
exactly, both with the 2015 gjh build used for the thesis and with the current
one installed automatically.

## Using the solver

```python
import sys
sys.path.insert(0, 'atrfu')
from TRF import TrustRegionSolver

# model: a Pyomo ConcreteModel whose black boxes are ExternalFunctions;
# efmap: {"block1": [ef1, ef2], "block2": [ef3], ...}, one entry per black box
solver = TrustRegionSolver(solver='ipopt', globalization_strategy=1,
                           multiple_black_boxes=1, restoration_strategy=1,
                           max_it=200)
result = solver.solve(model, efmap)   # dict: status, feasible, iterations, bb_evals, final_obj, ...
```

`problems/benchmark_problems.py` shows how problems are written. Every option,
with its default and description, is declared in `TrustRegionSolver.CONFIG`
(`atrfu/TRF.py`).

## Citation

Release 1.0.0 of this repository accompanies G. Hameed, *Rigorous Trust-Region Algorithms for Surrogate-based Grey-box Optimisation*, PhD thesis, University of Surrey, 2026; publications that use later releases are listed here as they appear. Please cite the release you used with the metadata in `CITATION.cff`; each release has its own Zenodo DOI, shown on the repository page.

## Licence and attribution

The code in this repository is released under the MIT licence (`LICENSE`), except the files derived from Pyomo listed below, which keep their 3-clause BSD notice.

`atrfu/PyomoInterface.py`, `Logger.py` and `readgjh.py` derive from the
trust-region code distributed with Pyomo (3-clause BSD licence, © 2017 National
Technology and Engineering Solutions of Sandia, LLC); their headers retain the
notice.
