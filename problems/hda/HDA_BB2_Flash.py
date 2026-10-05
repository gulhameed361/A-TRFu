"""
HDA Case Study - Black-Box 2: Flash Separator VLE (BB2)
=========================================================
Vapour-liquid equilibrium for the HDA reactor effluent stream.

Method:
  Rachford-Rice equation solved via Brent's method
  Wilson correlation for K-values (temperature and pressure dependent)
  Ideal enthalpy departure (Cp.dT) for both phases

Components: H2, CH4, Toluene, Benzene, Diphenyl
  (light gases H2/CH4 go predominantly to vapour,
   aromatics Tol/Benz/Diph go predominantly to liquid)

Black-box inputs  (w2): F_H2_in, F_CH4_in, F_Tol_in, F_Benz_in,
                        F_Diph_in, T_FL, P_FL
Black-box outputs (d2): F_H2_vap,  F_CH4_vap,  F_Tol_vap,
                        F_Benz_vap, F_Diph_vap,
                        F_H2_liq,  F_CH4_liq,  F_Tol_liq,
                        F_Benz_liq, F_Diph_liq,
                        H_vap, H_liq

Units:
  Flowrates   : mol/s
  Temperature : K
  Pressure    : Pa
  Enthalpy    : MJ/s

References:
  Wilson (1969) K-value correlation
  Rachford & Rice (1952) flash equation
  Smith, Van Ness, Abbott - Introduction to Chemical Engineering
  Thermodynamics, 7th ed.

Author: HDA Case Study for A-TRFu benchmarking
"""

import numpy as np
from scipy.optimize import brentq
from pyomo.environ import ExternalFunction, value

# =============================================================================
# Component critical properties and Wilson parameters
# Tc [K], Pc [Pa], omega (acentric factor)
# Order: H2, CH4, Toluene, Benzene, Diphenyl
# =============================================================================

COMPS = ['H2', 'CH4', 'Tol', 'Benz', 'Diph']

# Critical temperatures [K]
Tc = {
    'H2':   33.2,
    'CH4':  190.6,
    'Tol':  591.8,
    'Benz': 562.2,
    'Diph': 789.0,
}

# Critical pressures [Pa]
Pc = {
    'H2':   1.297e6,
    'CH4':  4.600e6,
    'Tol':  4.109e6,
    'Benz': 4.898e6,
    'Diph': 3.990e6,
}

# Acentric factors [-]
omega = {
    'H2':   -0.216,
    'CH4':   0.011,
    'Tol':   0.263,
    'Benz':  0.212,
    'Diph':  0.438,
}

# Heat capacities [J/(mol K)] - same as BB1 for consistency
Cp = {
    'H2':   29.1,
    'CH4':  35.7,
    'Tol':  103.7,
    'Benz': 82.4,
    'Diph': 165.0,
}

# Heats of formation [J/mol] - same as BB1
Hf = {
    'H2':   0.0,
    'CH4':  -74850.0,
    'Tol':  50170.0,
    'Benz': 82930.0,
    'Diph': 182000.0,
}

# Reference temperature [K]
T_ref = 298.15


# =============================================================================
# Wilson K-value correlation
# K_i = (Pc_i / P) * exp(5.373 * (1 + omega_i) * (1 - Tc_i/T))
# =============================================================================

def wilson_K(T, P):
    """
    Compute Wilson K-values for all components.

    Parameters
    ----------
    T : float  Temperature [K]
    P : float  Pressure [Pa]

    Returns
    -------
    K : dict {component: K_value}
    """
    K = {}
    for c in COMPS:
        K[c] = (Pc[c] / P) * np.exp(
            5.373 * (1.0 + omega[c]) * (1.0 - Tc[c] / T)
        )
    return K


# =============================================================================
# Rachford-Rice equation and solver
# =============================================================================

def rachford_rice(psi, z, K):
    """
    Rachford-Rice objective function.
    psi = vapour fraction V/F in [0, 1]

    RR(psi) = sum_i [ z_i * (K_i - 1) / (1 + psi*(K_i - 1)) ] = 0
    """
    rr = 0.0
    for c in COMPS:
        rr += z[c] * (K[c] - 1.0) / (1.0 + psi * (K[c] - 1.0))
    return rr


def solve_flash(F_total, z, K):
    """
    Solve Rachford-Rice for vapour fraction psi.

    Parameters
    ----------
    F_total : float  total molar flowrate [mol/s]
    z       : dict   {component: mole fraction}
    K       : dict   {component: K-value}

    Returns
    -------
    psi : float  vapour fraction [mol vap / mol total]
    """
    # Bracket limits to avoid singularities
    # psi in (psi_min, psi_max) where denominators stay positive
    psi_min = 1.0 / (1.0 - max(K.values())) + 1e-8
    psi_max = 1.0 / (1.0 - min(K.values())) - 1e-8
    psi_min = max(psi_min, 1e-8)
    psi_max = min(psi_max, 1.0 - 1e-8)

    # Check if system is all vapour or all liquid
    rr_low  = rachford_rice(1e-8,       z, K)
    rr_high = rachford_rice(1.0 - 1e-8, z, K)

    if rr_low <= 0.0:
        return 0.0   # all liquid
    if rr_high >= 0.0:
        return 1.0   # all vapour

    # Brent's method for root finding - robust and reliable
    try:
        psi = brentq(rachford_rice, psi_min, psi_max,
                     args=(z, K), xtol=1e-10, rtol=1e-10,
                     maxiter=200)
    except ValueError:
        # Fallback: bisect over full [0,1] if bracket fails
        psi = brentq(rachford_rice, 1e-8, 1.0 - 1e-8,
                     args=(z, K), xtol=1e-8, rtol=1e-8,
                     maxiter=200)
    return psi


# =============================================================================
# Enthalpy calculation [MJ/s]
# =============================================================================

def stream_enthalpy(F_dict, T):
    """
    Total stream enthalpy relative to T_ref.
    H = sum_i F_i * [Hf_i + Cp_i * (T - T_ref)]   [MJ/s]
    """
    H = sum(F * (Hf[c] + Cp[c] * (T - T_ref))
            for c, F in F_dict.items())
    return H * 1e-6


# =============================================================================
# Core flash simulator
# =============================================================================

def HDA_Flash_sim(F_H2_in, F_CH4_in, F_Tol_in, F_Benz_in,
                  F_Diph_in, T_FL, P_FL):
    """
    Simulate isothermal flash separator using Rachford-Rice / Wilson K-values.

    Parameters
    ----------
    F_H2_in   : float  H2 inlet   [mol/s]
    F_CH4_in  : float  CH4 inlet  [mol/s]
    F_Tol_in  : float  Toluene inlet [mol/s]
    F_Benz_in : float  Benzene inlet [mol/s]
    F_Diph_in : float  Diphenyl inlet [mol/s]
    T_FL      : float  Flash temperature [K]
    P_FL      : float  Flash pressure [Pa]

    Returns  (12 values)
    -------
    F_H2_vap, F_CH4_vap, F_Tol_vap, F_Benz_vap, F_Diph_vap  [mol/s]
    F_H2_liq, F_CH4_liq, F_Tol_liq, F_Benz_liq, F_Diph_liq  [mol/s]
    H_vap  [MJ/s]
    H_liq  [MJ/s]
    """
    F_in = {
        'H2':   max(float(F_H2_in),   1e-12),
        'CH4':  max(float(F_CH4_in),  1e-12),
        'Tol':  max(float(F_Tol_in),  1e-12),
        'Benz': max(float(F_Benz_in), 1e-12),
        'Diph': max(float(F_Diph_in), 1e-12),
    }
    T_FL = float(T_FL)
    P_FL = float(P_FL)

    F_total = sum(F_in.values())

    # Mole fractions
    z = {c: F_in[c] / F_total for c in COMPS}

    # Wilson K-values at (T_FL, P_FL)
    K = wilson_K(T_FL, P_FL)

    # Solve Rachford-Rice for vapour fraction psi
    psi = solve_flash(F_total, z, K)
    psi = np.clip(psi, 0.0, 1.0)

    # Phase compositions
    # y_i = z_i * K_i / (1 + psi*(K_i - 1))  (vapour mole fraction)
    # x_i = z_i / (1 + psi*(K_i - 1))         (liquid mole fraction)
    y = {}
    x = {}
    for c in COMPS:
        denom = 1.0 + psi * (K[c] - 1.0)
        y[c] = z[c] * K[c] / denom
        x[c] = z[c] / denom

    # Normalise to avoid tiny numerical drift
    y_sum = sum(y.values())
    x_sum = sum(x.values())
    y = {c: y[c] / y_sum for c in COMPS}
    x = {c: x[c] / x_sum for c in COMPS}

    # Molar flowrates per phase [mol/s]
    V = psi * F_total          # total vapour flow
    L = (1.0 - psi) * F_total  # total liquid flow

    F_vap = {c: y[c] * V for c in COMPS}
    F_liq = {c: x[c] * L for c in COMPS}

    # Phase enthalpies [MJ/s]
    H_vap = stream_enthalpy(F_vap, T_FL)
    H_liq = stream_enthalpy(F_liq, T_FL)

    return (
        float(F_vap['H2']),   float(F_vap['CH4']),
        float(F_vap['Tol']),  float(F_vap['Benz']),
        float(F_vap['Diph']),
        float(F_liq['H2']),   float(F_liq['CH4']),
        float(F_liq['Tol']),  float(F_liq['Benz']),
        float(F_liq['Diph']),
        float(H_vap),         float(H_liq),
    )


# =============================================================================
# Caching wrapper  (same pattern as CS3)
# =============================================================================

TRF_CACHE_FL = {}  # global cache for flash black-box

def TRF_blackbox_Flash(*args):
    """
    Cached wrapper around HDA_Flash_sim.

    Inputs : F_H2_in, F_CH4_in, F_Tol_in, F_Benz_in, F_Diph_in,
             T_FL, P_FL
    Returns: list of 12 floats
             [F_H2_vap .. F_Diph_vap, F_H2_liq .. F_Diph_liq,
              H_vap, H_liq]
    """
    args = [x for x in args if x is not None]
    args = args[0] if isinstance(args[0], list) else args
    inputs = [value(x) if hasattr(x, 'is_expression_type')
              else float(x) for x in args]

    key = tuple(round(float(x), 6) for x in inputs)

    if key in TRF_CACHE_FL:
        return TRF_CACHE_FL[key]

    (F_H2_in, F_CH4_in, F_Tol_in, F_Benz_in, F_Diph_in,
     T_FL, P_FL) = inputs

    result = list(HDA_Flash_sim(
        F_H2_in, F_CH4_in, F_Tol_in, F_Benz_in,
        F_Diph_in, T_FL, P_FL
    ))

    TRF_CACHE_FL[key] = result
    return result


# =============================================================================
# ExternalFunction declarations  (12 outputs -> 12 ExternalFunctions)
# All share the same cached simulator call
# =============================================================================

def _FL_F_H2_vap(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[0]

def _FL_F_CH4_vap(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[1]

def _FL_F_Tol_vap(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[2]

def _FL_F_Benz_vap(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[3]

def _FL_F_Diph_vap(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[4]

def _FL_F_H2_liq(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[5]

def _FL_F_CH4_liq(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[6]

def _FL_F_Tol_liq(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[7]

def _FL_F_Benz_liq(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[8]

def _FL_F_Diph_liq(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[9]

def _FL_H_vap(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[10]

def _FL_H_liq(a,b,c,d,e,f,g):
    return TRF_blackbox_Flash(a,b,c,d,e,f,g)[11]

# Pyomo ExternalFunction objects
FL_F_H2_vap   = ExternalFunction(_FL_F_H2_vap)    # [mol/s]
FL_F_CH4_vap  = ExternalFunction(_FL_F_CH4_vap)   # [mol/s]
FL_F_Tol_vap  = ExternalFunction(_FL_F_Tol_vap)   # [mol/s]
FL_F_Benz_vap = ExternalFunction(_FL_F_Benz_vap)  # [mol/s]
FL_F_Diph_vap = ExternalFunction(_FL_F_Diph_vap)  # [mol/s]
FL_F_H2_liq   = ExternalFunction(_FL_F_H2_liq)    # [mol/s]
FL_F_CH4_liq  = ExternalFunction(_FL_F_CH4_liq)   # [mol/s]
FL_F_Tol_liq  = ExternalFunction(_FL_F_Tol_liq)   # [mol/s]
FL_F_Benz_liq = ExternalFunction(_FL_F_Benz_liq)  # [mol/s]
FL_F_Diph_liq = ExternalFunction(_FL_F_Diph_liq)  # [mol/s]
FL_H_vap      = ExternalFunction(_FL_H_vap)        # [MJ/s]
FL_H_liq      = ExternalFunction(_FL_H_liq)        # [MJ/s]

# Convenience list for efmap
BB2_functions = [
    FL_F_H2_vap,  FL_F_CH4_vap,  FL_F_Tol_vap,
    FL_F_Benz_vap, FL_F_Diph_vap,
    FL_F_H2_liq,  FL_F_CH4_liq,  FL_F_Tol_liq,
    FL_F_Benz_liq, FL_F_Diph_liq,
    FL_H_vap, FL_H_liq,
]
