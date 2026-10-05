# -*- coding: utf-8 -*-
"""
BenchmarkProblems.py - grey-box benchmark set for the TRF solver variants.

Each builder returns (model, efmap, solver_kwargs):
  model         : fresh ConcreteModel (build a new one per run -- the solver
                  mutates config and the model is cloned internally)
  efmap         : {"block1": [bb1], ...} external-function map (one block per
                  BB sub-model; keys sort lexicographically -> deterministic
                  component order in the per-black-box infeasibility vector)
  solver_kwargs : problem-specific TrustRegionSolver settings

Black-box evaluation counts are reported by the solver itself (RUN SUMMARY:
"External BB evaluations" per block + TOTAL, plus BB-cache hit/miss), so the
problem builders no longer carry their own per-closure counters.

The 19 problems of thesis Chapter 7 (the 20th, HDA, is in problems/hda/):
  literature problems   : powell, colville, alkylation, williams_otto
  mechanism-targeted    : hetero_scale, noisy_bb, rosenbrock_bb, six_bb_chain
  standard-NLP battery  : infeasible_start, recycle_loop, multimodal,
                          ill_conditioned, active_set_churn, stiff_bb
  high-dimensional      : highdim_sep_10/20/40, highdim_coupled_10/20

@author: Gul Hameed
"""
import math

from pyomo.environ import (
    ConcreteModel, Var, Param, Constraint, Objective, ExternalFunction,
    RangeSet, maximize, minimize)


# =====================================================================
# Literature problems
# =====================================================================

def powell():
    """Powell's function, 4 vars, 2 BBs in the objective."""
    m = ConcreteModel()
    m.x1 = Var(initialize=3, bounds=(-4, 5))
    m.x2 = Var(initialize=-0.256837, bounds=(-4, 5))
    m.x3 = Var(initialize=0.517729, bounds=(-4, 5))
    m.x4 = Var(initialize=2.244261, bounds=(-4, 5))

    def blackbox1(a, b):
        return (a - 2*b)**4
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b):
        return 10*(a - b)**4
    bb2 = ExternalFunction(blackbox2)

    m.obj = Objective(
        expr=(m.x1 + 10*m.x2)**2
             + 5*(m.x3 - m.x4)**2
             + bb1(m.x2, m.x3)
             + bb2(m.x1, m.x4))

    efmap = {"block1": [bb1], "block2": [bb2]}
    return m, efmap, dict(scaling=0)


def colville():
    """Colville fourth problem variant, 5 vars, 4 BBs, 6 inequality
    constraints."""
    m = ConcreteModel()
    m.x1 = Var(initialize=77.99999922018861, bounds=(77, 102))
    m.x2 = Var(initialize=32.999999670073365, bounds=(32, 45))
    m.x3 = Var(initialize=29.99573956893894, bounds=(27, 45))
    m.x4 = Var(initialize=45.000000449581165, bounds=(27, 46))
    m.x5 = Var(initialize=36.77532790543838, bounds=(27, 45))

    def blackbox1(a, b):
        return 0.8357 * a * b
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b, c):
        return 0.00002584 * a * b - 0.00006663 * c * b
    bb2 = ExternalFunction(blackbox2)

    def blackbox3(a, b, c):
        return 2275.1327 * ((a * b)**(-1)) - 0.2668 * c * (b**(-1))
    bb3 = ExternalFunction(blackbox3)

    def blackbox4(a, b, c):
        return 1330.3294 * ((a * b)**(-1)) - 0.42 * c * (b**(-1))
    bb4 = ExternalFunction(blackbox4)

    m.c1 = Constraint(expr=bb2(m.x3, m.x5, m.x2) - 0.0000734*m.x1*m.x4 - 1 <= 0)
    m.c2 = Constraint(expr=0.000853007*m.x2*m.x5 + 0.00009395*m.x1*m.x4
                           - 0.00033085*m.x3*m.x5 - 1 <= 0)
    m.c3 = Constraint(expr=bb4(m.x2, m.x5, m.x1)
                           - 0.30586*((m.x2*m.x5)**(-1))*m.x3**2 - 1 <= 0)
    m.c4 = Constraint(expr=0.00024186*m.x2*m.x5 + 0.00010159*m.x1*m.x2
                           + 0.00007379*m.x3**2 - 1 <= 0)
    m.c5 = Constraint(expr=bb3(m.x3, m.x5, m.x1)
                           - 0.40584*(m.x5**(-1))*m.x4 - 1 <= 0)
    m.c6 = Constraint(expr=0.00029955*m.x3*m.x5 + 0.00007992*m.x1*m.x2
                           + 0.00012157*m.x3*m.x4 - 1 <= 0)

    m.obj = Objective(
        expr=5.3578*m.x3**2 + bb1(m.x1, m.x5) + 37.2392*m.x1,
        sense=minimize)

    efmap = {"block1": [bb1], "block2": [bb2], "block3": [bb3], "block4": [bb4]}
    return m, efmap, dict(scaling=1, trust_radius=1000, sample_radius=100)


def alkylation():
    """Alkylation process, 10 vars, 2 BBs, 7 constraints."""
    m = ConcreteModel()
    m.x = Var(
        RangeSet(0, 9),
        bounds={0: (0, 2000), 1: (0, 16000), 2: (0, 120), 3: (0, 5000),
                4: (0, 2000), 5: (85, 93), 6: (90, 95), 7: (5.69, 12),
                8: (1.2, 4), 9: (145, 162)},
        initialize={0: 1309.276241883202, 1: 6210.726557172016, 2: 120.0,
                    3: 2088.796312316891, 4: 1239.055259143405,
                    5: 92.99796451282054, 6: 92.66666666666667,
                    7: 5.69, 8: 3.090000000000005, 9: 145.0})

    def blackbox1(a):
        return 1.12 + (0.12167*a) - (0.0067*(a**2))
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a):
        return 86.35 + (1.098*a) - (0.038*(a**2))
    bb2 = ExternalFunction(blackbox2)

    m.c1 = Constraint(expr=m.x[3] == m.x[0] * bb1(m.x[7]))
    m.c2 = Constraint(expr=m.x[6] == bb2(m.x[7]) + (0.325*(m.x[5] - 89)))
    m.c3 = Constraint(expr=m.x[8] == 35.28 - (0.222*m.x[9]))
    m.c4 = Constraint(expr=m.x[9] == (3*m.x[6]) - 133)
    m.c5 = Constraint(expr=m.x[7]*m.x[0] == m.x[1] + m.x[4])
    m.c6 = Constraint(expr=m.x[4] == (1.22*m.x[3]) - m.x[0])
    m.c7 = Constraint(
        expr=(m.x[5]*m.x[3]*m.x[8]) + (m.x[5]*(1000*m.x[2])) == 98000*m.x[2])

    m.obj = Objective(
        expr=((0.063*m.x[3]*m.x[6]) - (5.04*m.x[0]) - (0.035*m.x[1])
              - (10*m.x[2]) - (3.36*m.x[4])),
        sense=maximize)

    efmap = {"block1": [bb1], "block2": [bb2]}
    return m, efmap, dict(scaling=0)


def williams_otto():
    """Williams-Otto reactor/recycle process, 3 BBs."""
    m = ConcreteModel()
    m.p = Param(default=50)

    m.V = Var(bounds=(0.03, 0.05), initialize=0.04)
    m.T = Var(bounds=(6.5, 6.8), initialize=6.8)
    m.Fp = Var(bounds=(3, 4.77), initialize=4.5)
    m.Fpurge = Var(bounds=(34, 36), initialize=35)
    m.Fg = Var(bounds=(2, 4), initialize=2.4822536092549936)
    m.Feff_sum = Var(initialize=299.99999702591543, bounds=(200, 400))
    m.Fa = Var(bounds=(9, 15), initialize=9.786572286966695)
    m.Fb = Var(bounds=(20, 40), initialize=32.195681322288294)
    m.n = Var(bounds=(0, 1), initialize=0.11944669149346067)

    m.Feff = Var(range(6), bounds=(0, 150))
    m.FR = Var(range(6), bounds=(0, 150))
    m.x = Var(range(6), bounds=(0, 1))

    Feff_vals = [17.78812335665588, 149.64474074144297, 2.9326822703152824,
                 111.50199731658753, 15.650199731658754, 2.4822536092549936]
    FR_vals = [15.663390873825776, 131.77017156048086, 2.58238307592459,
               98.18345264220837, 13.780835152500265, 2.185756628181784]
    x_vals = [0.059293745110001565, 0.49881580741654485, 0.009775607664629211,
              0.3716733280732514, 0.05216733295602931, 0.008274178779543617]
    for i in range(6):
        m.Feff[i] = Feff_vals[i]
        m.FR[i] = FR_vals[i]
        m.x[i] = x_vals[i]

    def blackbox1(a, b, c, d):
        return 5.9755e9 * math.exp(-120/a) * b * c * d * 50
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b, c, d):
        return 2.5962e12 * math.exp(-150/a) * b * c * d * 50
    bb2 = ExternalFunction(blackbox2)

    def blackbox3(a, b, c, d):
        return 9.6283e15 * math.exp(-200/a) * b * c * d * 50
    bb3 = ExternalFunction(blackbox3)

    m.obj = Objective(
        expr=(100 * ((2207*m.Fp) + (50*m.Fpurge) - (168*m.Fa) - (252*m.Fb)
                     - (2.22*m.Feff_sum) - (84*m.Fg) - (60*m.V*m.p))
              / (600*m.V*m.p)),
        sense=maximize)

    m.c4 = Constraint(expr=m.Feff[0] == m.Fa + m.FR[0]
                           - bb1(m.T, m.x[0], m.x[1], m.V))
    m.c5 = Constraint(expr=m.Feff[1] == m.Fb + m.FR[1]
                           - (bb1(m.T, m.x[0], m.x[1], m.V)
                              + bb2(m.T, m.x[1], m.x[2], m.V)))
    m.c6 = Constraint(expr=m.Feff[2] == m.FR[2]
                           + 2*bb1(m.T, m.x[0], m.x[1], m.V)
                           - 2*bb2(m.T, m.x[1], m.x[2], m.V)
                           - bb3(m.T, m.x[4], m.x[2], m.V))
    m.c7 = Constraint(expr=m.Feff[3] == m.FR[3]
                           + 2*bb2(m.T, m.x[1], m.x[2], m.V))
    m.c8 = Constraint(expr=m.Feff[4] == 0.1*m.FR[3]
                           + bb2(m.T, m.x[1], m.x[2], m.V)
                           - 0.5*bb3(m.T, m.x[4], m.x[2], m.V))
    m.c9 = Constraint(expr=m.Feff[5] == 1.5*bb3(m.T, m.x[4], m.x[2], m.V))
    m.c10 = Constraint(expr=m.Feff_sum == sum(m.Feff[i] for i in range(6)))
    m.c11 = Constraint(expr=m.Feff[0] == m.Feff_sum * m.x[0])
    m.c12 = Constraint(expr=m.Feff[1] == m.Feff_sum * m.x[1])
    m.c13 = Constraint(expr=m.Feff[2] == m.Feff_sum * m.x[2])
    m.c14 = Constraint(expr=m.Feff[3] == m.Feff_sum * m.x[3])
    m.c15 = Constraint(expr=m.Feff[4] == m.Feff_sum * m.x[4])
    m.c16 = Constraint(expr=m.Feff[5] == m.Feff_sum * m.x[5])
    m.c17 = Constraint(expr=m.Fg == m.Feff[5])
    m.c18 = Constraint(expr=m.Fp == m.Feff[4] - 0.1*m.Feff[3])
    m.c19 = Constraint(expr=m.Fpurge == m.n * (m.Feff[0] + m.Feff[1]
                                               + m.Feff[2] + 1.1*m.Feff[3]))
    m.c20 = Constraint(expr=m.FR[0] == (1 - m.n) * m.Feff[0])
    m.c21 = Constraint(expr=m.FR[1] == (1 - m.n) * m.Feff[1])
    m.c22 = Constraint(expr=m.FR[2] == (1 - m.n) * m.Feff[2])
    m.c23 = Constraint(expr=m.FR[3] == (1 - m.n) * m.Feff[3])
    m.c24 = Constraint(expr=m.FR[4] == (1 - m.n) * m.Feff[4])
    m.c25 = Constraint(expr=m.FR[5] == (1 - m.n) * m.Feff[5])

    efmap = {"block1": [bb1], "block2": [bb2], "block3": [bb3]}
    return m, efmap, dict(scaling=0)


# =====================================================================
# New problems (mechanism-targeted)
# =====================================================================

def hetero_scale():
    """3 BBs whose output magnitudes span six orders (1e-2, 1, 1e4).
    Unnormalised, BB3 dominates every aggregate infeasibility measure."""
    m = ConcreteModel()
    m.x1 = Var(initialize=1.5, bounds=(0.5, 3.0))
    m.x2 = Var(initialize=0.5, bounds=(0.1, 2.0))
    m.y1 = Var(initialize=0.0, bounds=(-1, 1))          # ~1e-2 scale
    m.y2 = Var(initialize=1.0, bounds=(-10, 10))        # ~1 scale
    m.y3 = Var(initialize=2e4, bounds=(0, 1e5))         # ~1e4 scale

    def blackbox1(a):
        return 0.01 * (a**2 - 2*a + 1.5)
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b):
        return a * b + 0.5 * b**2
    bb2 = ExternalFunction(blackbox2)

    def blackbox3(a, b):
        return 1.0e4 * (a + b**2)
    bb3 = ExternalFunction(blackbox3)

    m.c1 = Constraint(expr=m.y1 == bb1(m.x1))
    m.c2 = Constraint(expr=m.y2 == bb2(m.x1, m.x2))
    m.c3 = Constraint(expr=m.y3 == bb3(m.x1, m.x2))

    # well-scaled objective touching all three BB outputs
    m.obj = Objective(
        expr=(m.x1 - 2)**2 + (m.x2 - 1)**2
             + 100*m.y1 + 0.1*m.y2 + 1e-4*m.y3,
        sense=minimize)

    efmap = {"block1": [bb1], "block2": [bb2], "block3": [bb3]}
    return m, efmap, dict(scaling=0)


def noisy_bb():
    """One clean BB and one BB with a deterministic micro-oscillation
    (amplitude 1e-5, 'integrator noise'): the funnel must not contract below
    the oscillation amplitude."""
    m = ConcreteModel()
    m.x1 = Var(initialize=2.0, bounds=(0.5, 4.0))
    m.x2 = Var(initialize=1.0, bounds=(0.5, 4.0))
    m.y1 = Var(initialize=4.0, bounds=(0, 20))
    m.y2 = Var(initialize=2.0, bounds=(0, 20))

    def blackbox1(a, b):
        # smooth truth + deterministic high-frequency wiggle ~ solver noise
        return a**2 + 0.5*b + 1e-5 * math.sin(1000.0 * (a + b))
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b):
        return a * b
    bb2 = ExternalFunction(blackbox2)

    m.c1 = Constraint(expr=m.y1 == bb1(m.x1, m.x2))
    m.c2 = Constraint(expr=m.y2 == bb2(m.x1, m.x2))

    m.obj = Objective(
        expr=(m.y1 - 3.0)**2 + (m.y2 - 2.0)**2 + 0.1*(m.x1 - 1.5)**2,
        sense=minimize)

    efmap = {"block1": [bb1], "block2": [bb2]}
    return m, efmap, dict(scaling=0)


def rosenbrock_bb():
    """Constrained Rosenbrock valley with the curvature hidden in BBs.
    The banana valley forces step rejections on linear surrogates ->
    exercises the rejection->contraction anchor and the restoration phase."""
    m = ConcreteModel()
    m.x1 = Var(initialize=-1.2, bounds=(-2, 2))
    m.x2 = Var(initialize=1.0, bounds=(-1, 3))
    m.y1 = Var(initialize=1.44, bounds=(0, 8))
    m.y2 = Var(initialize=0.0, bounds=(-10, 10))

    def blackbox1(a):
        return a**2
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b):
        # curved coupling, = 0 along the valley b = a^2. NOTE: BB inputs must
        # be plain bounded VARIABLES (buildROM samples a box around them), so
        # the valley residual is computed inside the BB, not via an
        # expression argument.
        v = b - a**2
        return math.exp(0.5*a) * v - v
    bb2 = ExternalFunction(blackbox2)

    m.c1 = Constraint(expr=m.y1 == bb1(m.x1))
    m.c2 = Constraint(expr=m.y2 == bb2(m.x1, m.x2))

    # Rosenbrock through the BB: 100*(x2 - x1^2)^2 + (1-x1)^2
    m.obj = Objective(
        expr=100*(m.x2 - m.y1)**2 + (1 - m.x1)**2 + 0.5*m.y2**2,
        sense=minimize)

    efmap = {"block1": [bb1], "block2": [bb2]}
    return m, efmap, dict(scaling=0)


def six_bb_chain():
    """J=6 coupled BB sub-models sharing inputs in a chain. Stresses the
    funnel and the hypervolume machinery in 6 error dimensions."""
    m = ConcreteModel()
    m.x = Var(range(6), initialize=1.0, bounds=(0.2, 3.0))
    m.y = Var(range(6), initialize=1.0, bounds=(-50, 50))

    coeff = [1.0, 0.5, 2.0, 0.8, 1.5, 0.3]
    targets = [1.2, 0.9, 2.5, 1.1, 1.8, 0.6]

    def make_bb(i):
        c = coeff[i]
        def bb(a, b):
            return c * (a * b + 0.3 * a**2)
        return bb

    bbs = [ExternalFunction(make_bb(i)) for i in range(6)]

    def chain_rule(m, i):
        # each BB couples x[i] with its neighbour -> shared inputs across BBs
        return m.y[i] == bbs[i](m.x[i], m.x[(i + 1) % 6])
    m.chain = Constraint(range(6), rule=chain_rule)

    m.obj = Objective(
        expr=sum((m.y[i] - targets[i])**2 for i in range(6))
             + 0.1 * sum((m.x[i] - 1.0)**2 for i in range(6)),
        sense=minimize)

    efmap = {f"block{i+1}": [bbs[i]] for i in range(6)}
    return m, efmap, dict(scaling=0)


# =====================================================================
# Standard-NLP diagnostic battery

# Every problem below satisfies the standard NLP assumptions: f, g, h and the
# black-box truth t(.) are C2-smooth; the problem is feasible with a KKT point;
# the objective is bounded below; LICQ holds at the solution; variables are
# bounded (compact level sets). The DIFFICULTY comes from conditioning,
# nonlinearity, coupling, infeasible starts, multimodality and stiffness --
# never from non-smoothness or stochastic noise.
# =====================================================================

def infeasible_start():
    """3 smooth BBs; the PROBLEM is well-posed (feasible, smooth, regular) but
    the BB-output auxiliaries y are INITIALISED far from t(x), so theta_0 ~ 1e3.
    A fast stand-in for HDA's extreme-infeasibility start: stresses the
    restoration phase and the funnel switching test in seconds rather than
    minutes."""
    m = ConcreteModel()
    m.x1 = Var(initialize=2.0, bounds=(0.5, 3.0))
    m.x2 = Var(initialize=1.5, bounds=(0.5, 3.0))
    # y deliberately far from the true BB values (true ~O(1-10); start ~1e3)
    m.y1 = Var(initialize=800.0, bounds=(-50, 2000))
    m.y2 = Var(initialize=900.0, bounds=(-50, 2000))
    m.y3 = Var(initialize=700.0, bounds=(-50, 2000))

    def bb1(a, b):
        return a**2 + 0.5*b
    def bb2(a, b):
        return math.exp(0.4*a) + b**2
    def bb3(a, b):
        return a*b + 0.3*a**2
    f1, f2, f3 = (ExternalFunction(bb1), ExternalFunction(bb2),
                  ExternalFunction(bb3))
    m.c1 = Constraint(expr=m.y1 == f1(m.x1, m.x2))
    m.c2 = Constraint(expr=m.y2 == f2(m.x1, m.x2))
    m.c3 = Constraint(expr=m.y3 == f3(m.x1, m.x2))
    m.obj = Objective(
        expr=(m.x1 - 1.8)**2 + (m.x2 - 1.2)**2
             + 0.1*m.y1 + 0.05*m.y2 + 0.05*m.y3,
        sense=minimize)
    efmap = {"block1": [f1], "block2": [f2], "block3": [f3]}
    return m, efmap, dict(scaling=0)


def recycle_loop():
    """Smooth equality RECYCLE loop (HDA-style): total feed T = F + R, a BB
    conversion conv = 1 - exp(-0.3 T), and recycle R = (1-s) T (1-conv). The
    BB output feeds back into its own input through the equalities, so y is
    tightly pinned -- stresses restoration effectiveness and TRSP feasibility
    under coupling. Smooth and regular."""
    m = ConcreteModel()
    m.F = Var(initialize=2.0, bounds=(1.0, 5.0))      # fresh feed
    m.s = Var(initialize=0.5, bounds=(0.2, 0.8))      # purge split
    m.T = Var(initialize=4.0, bounds=(1.0, 25.0))     # total feed
    m.R = Var(initialize=2.0, bounds=(0.0, 20.0))     # recycle
    m.conv = Var(initialize=0.5, bounds=(0.0, 1.0))   # BB conversion

    def bb_conv(t):
        return 1.0 - math.exp(-0.3*t)
    fc = ExternalFunction(bb_conv)
    m.c1 = Constraint(expr=m.T == m.F + m.R)
    m.c2 = Constraint(expr=m.conv == fc(m.T))
    m.c3 = Constraint(expr=m.R == (1.0 - m.s) * m.T * (1.0 - m.conv))
    # maximise product (F*conv) minus recycle handling cost
    m.obj = Objective(expr=m.F * m.conv * 10.0 - 0.5 * m.R - 2.0 * m.F,
                      sense=maximize)
    efmap = {"block1": [fc]}
    return m, efmap, dict(scaling=0)


def multimodal():
    """Smooth DOUBLE-WELL BB (a^2 - 1)^2 (two minima at a = +/-1), slightly
    tilted by a linear term so the two local KKT points have different
    objective values. Probes which basin each globalization strategy selects
    (filter vs funnel). Fully smooth, bounded, regular."""
    m = ConcreteModel()
    m.x1 = Var(initialize=1.3, bounds=(-2.0, 2.0))
    m.x2 = Var(initialize=0.0, bounds=(-2.0, 2.0))
    m.y = Var(initialize=0.0, bounds=(-1.0, 30.0))

    def bb(a):
        return (a**2 - 1.0)**2
    fb = ExternalFunction(bb)
    m.c1 = Constraint(expr=m.y == fb(m.x1))
    # double well in y(x1); the +0.4*x1 tilt makes x1=-1 the lower basin
    m.obj = Objective(expr=m.y + 0.4*m.x1 + 0.1*(m.x2 - 0.5)**2,
                      sense=minimize)
    efmap = {"block1": [fb]}
    return m, efmap, dict(scaling=0)


def ill_conditioned():
    """Decision variables spanning six orders of magnitude (x1 ~ 1e-2,
    x2 ~ 1e3) with smooth BBs; the objective is well-scaled so the optimum is
    benign but the FD-ROM build and TR-box scaling must cope with the input
    ill-conditioning. Smooth, regular. Run with scaling=0 (bound-width
    scaling) -- it is the configuration designed to handle this."""
    m = ConcreteModel()
    m.x1 = Var(initialize=1e-2, bounds=(1e-3, 1e-1))   # small-scale input
    m.x2 = Var(initialize=1e3,  bounds=(1e2, 1e4))     # large-scale input
    m.y1 = Var(initialize=0.0, bounds=(-10, 10))
    m.y2 = Var(initialize=0.0, bounds=(-100, 100))

    def bb1(a):
        return 100.0 * a**2          # a~1e-2 -> ~1e-2
    def bb2(b):
        return 1e-3 * b              # b~1e3 -> ~1
    f1, f2 = ExternalFunction(bb1), ExternalFunction(bb2)
    m.c1 = Constraint(expr=m.y1 == f1(m.x1))
    m.c2 = Constraint(expr=m.y2 == f2(m.x2))
    m.obj = Objective(
        expr=(1e2*m.x1 - 5.0)**2 + (1e-3*m.x2 - 3.0)**2 + m.y1 + m.y2,
        sense=minimize)
    efmap = {"block1": [f1], "block2": [f2]}
    return m, efmap, dict(scaling=0)


def active_set_churn():
    """Smooth, BB-coupled INEQUALITIES arranged so several constraints are
    near-active and trade off as the iterate moves (the active set changes
    along the path). Probes the criticality LP's near-active filtering and the
    filter/funnel behaviour under a moving active set. LICQ holds at the
    solution (no redundant active constraints by construction)."""
    m = ConcreteModel()
    m.x1 = Var(initialize=0.5, bounds=(-2.0, 2.0))
    m.x2 = Var(initialize=0.5, bounds=(-2.0, 2.0))
    m.x3 = Var(initialize=0.5, bounds=(-2.0, 2.0))
    m.x4 = Var(initialize=0.5, bounds=(-2.0, 2.0))

    def bb1(a, b):
        return a**2 + b**2
    def bb2(a, b):
        return math.exp(0.5*a) + 0.5*b
    def bb3(a, b):
        return a*b + 0.25*a**2
    f1, f2, f3 = (ExternalFunction(bb1), ExternalFunction(bb2),
                  ExternalFunction(bb3))
    # three smooth BB-coupled inequalities + a linear one; the optimum sits
    # where two of them are active, and the path activates/deactivates them
    m.g1 = Constraint(expr=f1(m.x1, m.x2) <= 1.5)
    m.g2 = Constraint(expr=f2(m.x2, m.x3) <= 2.5)
    m.g3 = Constraint(expr=f3(m.x3, m.x4) <= 1.0)
    m.g4 = Constraint(expr=m.x1 + m.x2 + m.x3 + m.x4 <= 3.0)
    m.obj = Objective(
        expr=-(m.x1 + 0.8*m.x2 + 0.6*m.x3 + 0.4*m.x4)
             + 0.1*(m.x1**2 + m.x2**2 + m.x3**2 + m.x4**2),
        sense=minimize)
    efmap = {"block1": [f1], "block2": [f2], "block3": [f3]}
    return m, efmap, dict(scaling=0)


def stiff_bb():
    """Smooth but STIFF BB: y = exp(5 x1) varies by ~4 orders of magnitude
    over x1 in [0,2] (gradient up to 5 e^10, large but finite -> still C2 and
    regular). A linear surrogate is accurate only over a tiny step, so this
    stresses the ROM finite-difference step-size guards and the trust-region
    response to a fast-changing BB (an HDA-reactor-runaway in miniature).
    Well-posed: unique minimiser at exp(5 x1) = 50."""
    m = ConcreteModel()
    m.x1 = Var(initialize=0.4, bounds=(0.0, 2.0))
    m.x2 = Var(initialize=1.0, bounds=(0.0, 3.0))
    m.y1 = Var(initialize=0.0, bounds=(0.0, 1e5))
    m.y2 = Var(initialize=0.0, bounds=(-10, 50))

    def bb1(a):
        return math.exp(5.0*a)        # stiff
    def bb2(a, b):
        return a*b + 0.5*b            # mild companion
    f1, f2 = ExternalFunction(bb1), ExternalFunction(bb2)
    m.c1 = Constraint(expr=m.y1 == f1(m.x1))
    m.c2 = Constraint(expr=m.y2 == f2(m.x1, m.x2))
    m.obj = Objective(expr=(m.y1 - 50.0)**2 * 1e-3 + (m.y2 - 2.0)**2,
                      sense=minimize)
    efmap = {"block1": [f1], "block2": [f2]}
    return m, efmap, dict(scaling=0)


# =====================================================================
# High-dimensional scalable problems (standard NLP assumptions hold).
# Difficulty is DIMENSION, not pathology: smooth C2 BBs, feasible, regular,
# bounded. Two distinct stressors:
#   highdim_separable(n) : n decision vars, n one-input BBs -> stresses the
#       infeasibility vector length / HV box dimension / model size, with a
#       well-conditioned separable objective (benign optimum).
#   highdim_coupled(n)   : n decision vars, 3 BBs each depending on ALL n
#       inputs -> stresses the ROM finite-difference build cost (3*n BB calls
#       per rebuild) and the dense criticality LP.
# =====================================================================

def highdim_separable(n):
    """n vars, n smooth one-input BBs, well-conditioned separable objective.
    Block names zero-padded (block00..) so the sorted component order is the
    natural one even for n >= 10. Minimiser is benign and unique."""
    m = ConcreteModel()
    m.x = Var(range(n), bounds=(-5.0, 5.0), initialize=2.0)
    m.y = Var(range(n), bounds=(-50.0, 200.0), initialize=0.0)
    targets = [1.0 + (i % 3) * 0.5 for i in range(n)]   # deterministic, varied

    def make_bb(i):
        def bb(a):
            return math.exp(0.1 * a) - 1.0 + 0.5 * a**2   # smooth, convex-ish
        return bb
    bbs = [ExternalFunction(make_bb(i)) for i in range(n)]

    def link(m, i):
        return m.y[i] == bbs[i](m.x[i])
    m.link = Constraint(range(n), rule=link)
    m.obj = Objective(
        expr=sum((m.x[i] - targets[i])**2 for i in range(n))
             + 0.1 * sum(m.y[i] for i in range(n)),
        sense=minimize)
    efmap = {f"block{i:02d}": [bbs[i]] for i in range(n)}
    return m, efmap, dict(scaling=0)


def highdim_coupled(n):
    """n vars, 3 smooth BBs each depending on ALL n inputs. The ROM build
    perturbs every input of every BB output, so each rebuild costs ~3*n true
    BB evaluations -- this is the finite-difference-build cost stressor.
    Smooth, feasible, regular, bounded."""
    m = ConcreteModel()
    m.x = Var(range(n), bounds=(-3.0, 3.0), initialize=0.5)
    m.y = Var(range(3), bounds=(-1e4, 1e4), initialize=0.0)
    targets = [1.0 + 0.3 * ((i * 7) % 5) for i in range(n)]  # deterministic

    def bb0(*a):
        return sum(0.1 * v**2 for v in a)                      # smooth bowl
    def bb1(*a):
        return sum(math.exp(0.05 * v) for v in a)              # smooth
    def bb2(*a):
        return sum(0.1 * a[i] * a[(i + 1) % len(a)] for i in range(len(a)))
    f0 = ExternalFunction(bb0)
    f1 = ExternalFunction(bb1)
    f2 = ExternalFunction(bb2)
    xs = [m.x[i] for i in range(n)]
    m.c0 = Constraint(expr=m.y[0] == f0(*xs))
    m.c1 = Constraint(expr=m.y[1] == f1(*xs))
    m.c2 = Constraint(expr=m.y[2] == f2(*xs))
    m.obj = Objective(
        expr=sum((m.x[i] - targets[i])**2 for i in range(n))
             + m.y[0] + 0.01 * m.y[1] + 0.1 * m.y[2],
        sense=minimize)
    efmap = {"block0": [f0], "block1": [f1], "block2": [f2]}
    return m, efmap, dict(scaling=0)


# Registry: name -> (builder, short description)
from functools import partial as _partial

PROBLEMS = {
    'powell':        (powell,        'literature: 4 vars, 2 BBs in objective'),
    'colville':      (colville,      'literature: 5 vars, 4 BBs, 6 ineq'),
    'alkylation':    (alkylation,    'literature: 10 vars, 2 BBs, 7 cons'),
    'williams_otto': (williams_otto, 'literature: WO process, 3 BBs, recycle'),
    'hetero_scale':  (hetero_scale,  'new: BB scales 1e-2..1e4 (normalization)'),
    'noisy_bb':      (noisy_bb,      'new: 1e-5 oscillation (BB noise)'),
    'rosenbrock_bb': (rosenbrock_bb, 'new: BB banana valley (rejections/restoration)'),
    'six_bb_chain':  (six_bb_chain,  'new: J=6 chain (HV in 6 dimensions)'),
    # --- standard-NLP diagnostic battery (smooth, feasible, regular) ---
    'infeasible_start':  (infeasible_start,  'NLP: theta_0~1e3 start (fast HDA-surrogate)'),
    'recycle_loop':      (recycle_loop,      'NLP: smooth equality recycle (restoration)'),
    'multimodal':        (multimodal,        'NLP: double-well BB (basin selection)'),
    'ill_conditioned':   (ill_conditioned,   'NLP: inputs 1e-3..1e3 (scaling/FD ROM)'),
    'active_set_churn':  (active_set_churn,  'NLP: BB-coupled ineqs (moving active set)'),
    'stiff_bb':          (stiff_bb,          'NLP: exp(5x) stiff BB (ROM step guards)'),
    # --- high-dimensional, scalable (standard NLP) ---
    'highdim_sep_10':     (_partial(highdim_separable, 10), 'HD: 10 vars, 10 one-input BBs'),
    'highdim_sep_20':     (_partial(highdim_separable, 20), 'HD: 20 vars, 20 one-input BBs'),
    'highdim_sep_40':     (_partial(highdim_separable, 40), 'HD: 40 vars, 40 one-input BBs'),
    'highdim_coupled_10': (_partial(highdim_coupled, 10),   'HD: 10 vars, 3 BBs on all inputs (ROM cost)'),
    'highdim_coupled_20': (_partial(highdim_coupled, 20),   'HD: 20 vars, 3 BBs on all inputs (ROM cost)'),
}
