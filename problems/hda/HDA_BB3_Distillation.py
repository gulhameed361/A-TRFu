"""
HDA Case Study - Black-Box 3: Distillation Train FUG Shortcut (BB3)
=====================================================================
Two-column distillation train for aromatics separation:

  T2 - Benzene Column : separates Benzene (distillate) from Toluene+Diphenyl
  T3 - Toluene Column : separates Toluene (distillate, recycle)
                        from Diphenyl (bottoms, purge)

Method: Fenske-Underwood-Gilliland (FUG) shortcut
  Fenske  -> minimum number of stages (N_min)
  Underwood -> minimum reflux ratio (RR_min)
  Gilliland -> actual stages from (RR - RR_min) / (RR + 1)
  Kremser  -> feed stage location

Condenser/reboiler duties from overall energy balance using
latent heat approximation.

Black-box inputs  (w3): F_Benz_in, F_Tol_in, F_Diph_in,
                        RR2, RR3
Black-box outputs (d3): F_Benz_product, F_Tol_recycle,
                        F_Diph_out, Q_T2, Q_T3

Units:
  Flowrates : mol/s
  RR        : dimensionless
  Duties    : MJ/s

References:
  Fenske (1932), Underwood (1948), Gilliland (1940)
  Seader & Henley, Separation Process Principles, 3rd ed.
  Douglas (1988), Conceptual Design of Chemical Processes

Author: HDA Case Study for A-TRFu benchmarking
"""

import numpy as np
from scipy.optimize import brentq
from pyomo.environ import ExternalFunction, value

# =============================================================================
# Component thermodynamic properties for distillation
# =============================================================================

COMPS_AROM = ['Benz', 'Tol', 'Diph']  # components in distillation feed

# Normal boiling points [K]
Tb = {
    'Benz': 353.2,
    'Tol':  383.8,
    'Diph': 528.0,
}

# Critical properties (same as BB2 for consistency)
Tc = {
    'Benz': 562.2,
    'Tol':  591.8,
    'Diph': 789.0,
}

Pc = {
    'Benz': 4.898e6,
    'Tol':  4.109e6,
    'Diph': 3.990e6,
}

omega = {
    'Benz':  0.212,
    'Tol':   0.263,
    'Diph':  0.438,
}

# Latent heats of vaporisation at normal boiling point [J/mol]
# Used for condenser/reboiler duty estimates
lambda_vap = {
    'Benz': 30720.0,
    'Tol':  33180.0,
    'Diph': 48500.0,
}

# Average latent heat for mixture approximation [J/mol]
lambda_avg = {
    'T2': 32000.0,   # benzene/toluene column
    'T3': 40000.0,   # toluene/diphenyl column
}

# Column operating pressures [Pa]
P_col = {
    'T2': 1.2 * 101325,   # slightly above atmospheric
    'T3': 1.0 * 101325,   # atmospheric
}

# Reference temperature for Wilson K-values in columns
# Approximate bubble point temperatures
T_col = {
    'T2': 365.0,   # K  (benzene/toluene system)
    'T3': 420.0,   # K  (toluene/diphenyl system)
}

# Minimum recovery fractions to avoid log(0) in Fenske
REC_MIN = 1e-6
REC_MAX = 1.0 - 1e-6


# =============================================================================
# Relative volatility from Wilson K-values
# =============================================================================

def wilson_K_col(comp, T, P):
    """Wilson K-value for a single component."""
    return (Pc[comp] / P) * np.exp(
        5.373 * (1.0 + omega[comp]) * (1.0 - Tc[comp] / T)
    )


def relative_volatility(comps, T, P, heavy_key):
    """
    Compute relative volatilities alpha_i = K_i / K_heavy_key.

    Parameters
    ----------
    comps     : list of component names
    T         : float  temperature [K]
    P         : float  pressure [Pa]
    heavy_key : str    heavy key component name

    Returns
    -------
    alpha : dict {comp: alpha_i}
    """
    K = {c: wilson_K_col(c, T, P) for c in comps}
    K_hk = K[heavy_key]
    alpha = {c: K[c] / K_hk for c in comps}
    return alpha


# =============================================================================
# Fenske equation: minimum stages
# =============================================================================

def fenske(alpha_lk, x_lk_D, x_hk_D, x_lk_B, x_hk_B):
    """
    Fenske equation for minimum number of theoretical stages.

    N_min = log[(x_lk_D/x_hk_D) * (x_hk_B/x_lk_B)] / log(alpha_lk)

    Parameters
    ----------
    alpha_lk : float  relative volatility of light key vs heavy key
    x_lk_D   : float  light key mole fraction in distillate
    x_hk_D   : float  heavy key mole fraction in distillate
    x_lk_B   : float  light key mole fraction in bottoms
    x_hk_B   : float  heavy key mole fraction in bottoms

    Returns
    -------
    N_min : float  minimum theoretical stages
    """
    # Clip to avoid log(0)
    ratio_D = max(x_lk_D, REC_MIN) / max(x_hk_D, REC_MIN)
    ratio_B = max(x_hk_B, REC_MIN) / max(x_lk_B, REC_MIN)
    N_min = np.log(ratio_D * ratio_B) / np.log(max(alpha_lk, 1.001))
    return max(N_min, 1.0)


# =============================================================================
# Underwood equation: minimum reflux ratio
# =============================================================================

def underwood(alpha, z_feed, q, comps):
    """
    Underwood equation for minimum reflux ratio.

    Solves sum_i [alpha_i * z_i / (alpha_i - theta)] = 1 - q
    for theta between alpha of heavy key and light key,
    then RR_min = sum_i [alpha_i * x_i_D / (alpha_i - theta)] - 1

    Parameters
    ----------
    alpha  : dict {comp: relative volatility}
    z_feed : dict {comp: feed mole fraction}
    q      : float  feed quality (1 = saturated liquid)
    comps  : list   component names

    Returns
    -------
    RR_min : float  minimum reflux ratio
    """
    # Sort components by volatility
    alpha_vals = [alpha[c] for c in comps]
    alpha_lk   = max(alpha_vals)   # light key has highest alpha
    alpha_hk   = min(alpha_vals)   # heavy key has lowest alpha (=1 by def)

    # Underwood root lies between alpha_hk and alpha_lk
    def underwood_eq(theta):
        return sum(alpha[c] * z_feed[c] / (alpha[c] - theta)
                   for c in comps) - (1.0 - q)

    # Find theta bracket
    eps = 1e-6
    lo = alpha_hk + eps
    hi = alpha_lk - eps

    # Guard: if function doesn't bracket, return reasonable default
    try:
        f_lo = underwood_eq(lo)
        f_hi = underwood_eq(hi)
        if f_lo * f_hi > 0:
            return 1.2   # fallback minimum reflux
        theta = brentq(underwood_eq, lo, hi,
                       xtol=1e-8, rtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return 1.2   # fallback

    # Minimum reflux from Underwood equation (using distillate compositions)
    # Approximate: use feed compositions scaled by split fractions
    # for a saturated liquid feed (q=1), RR_min typically ~1.0-1.5
    RR_min = sum(alpha[c] * z_feed[c] / (alpha[c] - theta)
                 for c in comps) - 1.0

    return max(float(RR_min), 0.5)


# =============================================================================
# Gilliland correlation: actual stages from RR and RR_min
# =============================================================================

def gilliland(N_min, RR, RR_min):
    """
    Gilliland correlation for actual number of theoretical stages.

    X = (RR - RR_min) / (RR + 1)
    Y = (N - N_min) / (N + 1)
    Molokanov approximation:
      Y = 1 - exp[(1 + 54.4*X)/(11 + 117.2*X) * (X-1)/X^0.5]

    Parameters
    ----------
    N_min  : float  minimum stages from Fenske
    RR     : float  actual reflux ratio
    RR_min : float  minimum reflux ratio from Underwood

    Returns
    -------
    N : float  actual theoretical stages
    """
    RR = max(RR, RR_min * 1.05)  # ensure RR > RR_min
    X = (RR - RR_min) / (RR + 1.0)
    X = np.clip(X, 1e-6, 1.0 - 1e-6)

    # Molokanov (1972) approximation to Gilliland chart
    Y = 1.0 - np.exp(
        ((1.0 + 54.4 * X) / (11.0 + 117.2 * X)) * ((X - 1.0) / np.sqrt(X))
    )
    Y = np.clip(Y, 0.0, 1.0 - 1e-6)

    # N from Y = (N - N_min)/(N + 1)
    N = (N_min + Y) / (1.0 - Y)
    return max(N, N_min + 1.0)


# =============================================================================
# Column energy balance: condenser + reboiler duties
# =============================================================================

def column_duties(D, B, RR, col_key):
    """
    Estimate condenser and total reboiler duty.

    Q_condenser = (RR + 1) * D * lambda_avg   [J/s]
    Q_reboiler  = Q_condenser - (D + B)*h_approx  ~ Q_condenser (simplified)

    For algorithmic benchmarking purposes we use:
      Q_col = (RR + 1) * D * lambda_avg  [MJ/s]
    This captures the key trade-off: higher RR -> higher duty.

    Parameters
    ----------
    D       : float  distillate molar flowrate [mol/s]
    B       : float  bottoms molar flowrate    [mol/s]
    RR      : float  reflux ratio
    col_key : str    'T2' or 'T3' (selects lambda_avg)

    Returns
    -------
    Q_col : float  column duty [MJ/s]
    """
    Q = (RR + 1.0) * D * lambda_avg[col_key]
    return float(Q) * 1e-6   # J/s -> MJ/s


# =============================================================================
# Single column FUG calculation
# =============================================================================

def fug_column(F_feed, comps, light_key, heavy_key,
               rec_lk, rec_hk, RR, col_key, T_op, P_op):
    """
    FUG shortcut for one distillation column.

    Parameters
    ----------
    F_feed    : dict {comp: flowrate [mol/s]}  feed to column
    comps     : list  all components present
    light_key : str   light key component
    heavy_key : str   heavy key component
    rec_lk    : float recovery of light key in distillate [0,1]
    rec_hk    : float recovery of heavy key in bottoms    [0,1]
    RR        : float actual reflux ratio
    col_key   : str   'T2' or 'T3' for property lookup
    T_op      : float operating temperature [K]
    P_op      : float operating pressure [Pa]

    Returns
    -------
    F_dist  : dict {comp: distillate flowrate [mol/s]}
    F_bot   : dict {comp: bottoms flowrate [mol/s]}
    N       : float actual theoretical stages
    RR_min  : float minimum reflux ratio
    Q_col   : float column duty [MJ/s]
    """
    F_total = sum(F_feed.values())
    if F_total < 1e-10:
        zero = {c: 0.0 for c in comps}
        return zero, zero, 1.0, 1.0, 0.0

    z = {c: F_feed[c] / F_total for c in comps}

    # Relative volatilities
    alpha = relative_volatility(comps, T_op, P_op, heavy_key)

    # Light and heavy key splits
    F_lk_D = rec_lk  * F_feed[light_key]
    F_hk_D = (1.0 - rec_hk) * F_feed[heavy_key]
    F_lk_B = (1.0 - rec_lk) * F_feed[light_key]
    F_hk_B = rec_hk  * F_feed[heavy_key]

    # Non-key components: distributed by Fenske (use geometric mean alpha)
    # lighter than lk -> go to distillate
    # heavier than hk -> go to bottoms
    # between lk/hk -> distribute by alpha ratio
    alpha_lk = alpha[light_key]
    alpha_hk = alpha[heavy_key]   # = 1 by definition

    F_dist = {light_key: F_lk_D, heavy_key: F_hk_D}
    F_bot  = {light_key: F_lk_B, heavy_key: F_hk_B}

    for c in comps:
        if c in (light_key, heavy_key):
            continue
        alpha_c = alpha[c]
        if alpha_c >= alpha_lk:
            # lighter than lk: essentially all to distillate
            F_dist[c] = F_feed[c] * REC_MAX
            F_bot[c]  = F_feed[c] * REC_MIN
        elif alpha_c <= alpha_hk:
            # heavier than hk: essentially all to bottoms
            F_dist[c] = F_feed[c] * REC_MIN
            F_bot[c]  = F_feed[c] * REC_MAX
        else:
            # intermediate: distribute proportionally to alpha
            frac_dist = (alpha_c - alpha_hk) / (alpha_lk - alpha_hk)
            frac_dist = np.clip(frac_dist, REC_MIN, REC_MAX)
            F_dist[c] = F_feed[c] * frac_dist
            F_bot[c]  = F_feed[c] * (1.0 - frac_dist)

    # Total distillate and bottoms flowrates
    D = sum(F_dist.values())
    B = sum(F_bot.values())

    if D < 1e-10 or B < 1e-10:
        N_stages = 2.0
        RR_min   = 1.0
        Q_col    = column_duties(max(D, 1e-10), max(B, 1e-10), RR, col_key)
        return F_dist, F_bot, N_stages, RR_min, Q_col

    # Compositions for Fenske
    x_lk_D = F_dist[light_key] / D
    x_hk_D = F_dist[heavy_key] / D
    x_lk_B = F_bot[light_key]  / B
    x_hk_B = F_bot[heavy_key]  / B

    # Fenske: N_min
    N_min = fenske(alpha_lk, x_lk_D, x_hk_D, x_lk_B, x_hk_B)

    # Underwood: RR_min (saturated liquid feed q=1)
    RR_min = underwood(alpha, z, q=1.0, comps=comps)

    # Gilliland: actual stages
    N_stages = gilliland(N_min, RR, RR_min)

    # Column duty
    Q_col = column_duties(D, B, RR, col_key)

    return F_dist, F_bot, N_stages, RR_min, Q_col


# =============================================================================
# Core distillation train simulator
# =============================================================================

def HDA_Distillation_sim(F_Benz_in, F_Tol_in, F_Diph_in, RR2, RR3):
    """
    Simulate the two-column HDA distillation train.

    T2: Benzene column
        Light key = Benzene, Heavy key = Toluene
        Distillate = benzene product
        Bottoms    = toluene + diphenyl -> T3

    T3: Toluene column
        Light key = Toluene, Heavy key = Diphenyl
        Distillate = toluene recycle -> M1
        Bottoms    = diphenyl byproduct -> exit

    Parameters
    ----------
    F_Benz_in : float  Benzene feed flowrate [mol/s]
    F_Tol_in  : float  Toluene feed flowrate [mol/s]
    F_Diph_in : float  Diphenyl feed flowrate [mol/s]
    RR2       : float  Reflux ratio for T2 (benzene column)
    RR3       : float  Reflux ratio for T3 (toluene column)

    Returns
    -------
    F_Benz_product : float  benzene product flowrate [mol/s]
    F_Tol_recycle  : float  toluene recycle flowrate [mol/s]
    F_Diph_out     : float  diphenyl bottoms flowrate [mol/s]
    Q_T2           : float  T2 column duty [MJ/s]
    Q_T3           : float  T3 column duty [MJ/s]
    """
    # Guard inputs
    F_Benz_in = max(float(F_Benz_in), 1e-10)
    F_Tol_in  = max(float(F_Tol_in),  1e-10)
    F_Diph_in = max(float(F_Diph_in), 1e-10)
    RR2 = max(float(RR2), 1.05)
    RR3 = max(float(RR3), 1.05)

    # -----------------------------------------------------------------------
    # T2: Benzene column
    # Feed: Benz + Tol + Diph
    # LK = Benz (99.7% recovery to distillate)
    # HK = Tol  (99.5% recovery to bottoms)
    # -----------------------------------------------------------------------
    feed_T2 = {
        'Benz': F_Benz_in,
        'Tol':  F_Tol_in,
        'Diph': F_Diph_in,
    }

    F_dist_T2, F_bot_T2, N_T2, RR_min_T2, Q_T2 = fug_column(
        F_feed    = feed_T2,
        comps     = COMPS_AROM,
        light_key = 'Benz',
        heavy_key = 'Tol',
        rec_lk    = 0.997,   # 99.7% benzene to distillate (purity spec)
        rec_hk    = 0.995,   # 99.5% toluene to bottoms
        RR        = RR2,
        col_key   = 'T2',
        T_op      = T_col['T2'],
        P_op      = P_col['T2'],
    )

    # Benzene product = distillate of T2
    F_Benz_product = F_dist_T2['Benz']

    # -----------------------------------------------------------------------
    # T3: Toluene column
    # Feed: bottoms from T2 (Tol + Diph + trace Benz)
    # LK = Tol  (99.5% recovery to distillate -> recycle)
    # HK = Diph (99.9% recovery to bottoms -> exit)
    # -----------------------------------------------------------------------
    feed_T3 = {
        'Benz': F_bot_T2.get('Benz', 1e-10),
        'Tol':  F_bot_T2.get('Tol',  1e-10),
        'Diph': F_bot_T2.get('Diph', 1e-10),
    }

    F_dist_T3, F_bot_T3, N_T3, RR_min_T3, Q_T3 = fug_column(
        F_feed    = feed_T3,
        comps     = COMPS_AROM,
        light_key = 'Tol',
        heavy_key = 'Diph',
        rec_lk    = 0.995,   # 99.5% toluene to distillate
        rec_hk    = 0.999,   # 99.9% diphenyl to bottoms
        RR        = RR3,
        col_key   = 'T3',
        T_op      = T_col['T3'],
        P_op      = P_col['T3'],
    )

    # Toluene recycle = distillate of T3
    F_Tol_recycle = F_dist_T3['Tol']

    # Diphenyl byproduct = bottoms of T3
    F_Diph_out = F_bot_T3['Diph']

    return (
        float(F_Benz_product),
        float(F_Tol_recycle),
        float(F_Diph_out),
        float(Q_T2),
        float(Q_T3),
    )


# =============================================================================
# Caching wrapper  (same pattern as CS3 / BB1 / BB2)
# =============================================================================

TRF_CACHE_DIST = {}  # global cache for distillation black-box

def TRF_blackbox_Distillation(*args):
    """
    Cached wrapper around HDA_Distillation_sim.

    Inputs : F_Benz_in, F_Tol_in, F_Diph_in, RR2, RR3
    Returns: list of 5 floats
             [F_Benz_product, F_Tol_recycle, F_Diph_out, Q_T2, Q_T3]
    """
    args = [x for x in args if x is not None]
    args = args[0] if isinstance(args[0], list) else args
    inputs = [value(x) if hasattr(x, 'is_expression_type')
              else float(x) for x in args]

    key = tuple(round(float(x), 6) for x in inputs)

    if key in TRF_CACHE_DIST:
        return TRF_CACHE_DIST[key]

    F_Benz_in, F_Tol_in, F_Diph_in, RR2, RR3 = inputs

    result = list(HDA_Distillation_sim(
        F_Benz_in, F_Tol_in, F_Diph_in, RR2, RR3
    ))

    TRF_CACHE_DIST[key] = result
    return result


# =============================================================================
# ExternalFunction declarations  (5 outputs -> 5 ExternalFunctions)
# All share the same cached simulator call
# =============================================================================

def _Dist_F_Benz_product(a, b, c, d, e):
    return TRF_blackbox_Distillation(a, b, c, d, e)[0]

def _Dist_F_Tol_recycle(a, b, c, d, e):
    return TRF_blackbox_Distillation(a, b, c, d, e)[1]

def _Dist_F_Diph_out(a, b, c, d, e):
    return TRF_blackbox_Distillation(a, b, c, d, e)[2]

def _Dist_Q_T2(a, b, c, d, e):
    return TRF_blackbox_Distillation(a, b, c, d, e)[3]

def _Dist_Q_T3(a, b, c, d, e):
    return TRF_blackbox_Distillation(a, b, c, d, e)[4]

# Pyomo ExternalFunction objects
Dist_F_Benz_product = ExternalFunction(_Dist_F_Benz_product)  # [mol/s]
Dist_F_Tol_recycle  = ExternalFunction(_Dist_F_Tol_recycle)   # [mol/s]
Dist_F_Diph_out     = ExternalFunction(_Dist_F_Diph_out)      # [mol/s]
Dist_Q_T2           = ExternalFunction(_Dist_Q_T2)             # [MJ/s]
Dist_Q_T3           = ExternalFunction(_Dist_Q_T3)             # [MJ/s]

# Convenience list for efmap
BB3_functions = [
    Dist_F_Benz_product,
    Dist_F_Tol_recycle,
    Dist_F_Diph_out,
    Dist_Q_T2,
    Dist_Q_T3,
]
