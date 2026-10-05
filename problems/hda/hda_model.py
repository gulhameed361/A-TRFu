"""
HDA (hydrodealkylation of toluene) case study -- problem 20 of thesis Chapter 7.

Glass-box flowsheet (mixer, furnace, splitter, compressor, stabiliser, recycle
closures, purity and production specifications, total annualised cost) around
three black boxes: the PFR reactor (HDA_BB1_Reactor), the flash
(HDA_BB2_Flash) and the distillation train (HDA_BB3_Distillation).

build_hda() returns (model, efmap, solver_kwargs) in the same convention as
problems/benchmark_problems.py. The initial point is made consistent by one
pass through the three black boxes at the nominal design (the "pre-solve").
"""
from pyomo.environ import (
    ConcreteModel, Var, Param, Constraint, Objective, minimize
)

from HDA_BB1_Reactor import (
    TRF_CACHE_R, BB1_functions,
    HDA_Reactor_sim,
    R_F_H2_out, R_F_CH4_out, R_F_Tol_out,
    R_F_Benz_out, R_F_Diph_out, R_T_out, R_H_out
)
from HDA_BB2_Flash import (
    TRF_CACHE_FL, BB2_functions,
    HDA_Flash_sim,
    FL_F_H2_vap,  FL_F_CH4_vap, FL_F_Tol_vap,
    FL_F_Benz_vap, FL_F_Diph_vap,
    FL_F_H2_liq,  FL_F_CH4_liq, FL_F_Tol_liq,
    FL_F_Benz_liq, FL_F_Diph_liq,
    FL_H_vap, FL_H_liq
)
from HDA_BB3_Distillation import (
    TRF_CACHE_DIST, BB3_functions,
    HDA_Distillation_sim,
    Dist_F_Benz_product, Dist_F_Tol_recycle,
    Dist_F_Diph_out, Dist_Q_T2, Dist_Q_T3
)

# =============================================================================
# Plant / costing parameters
# =============================================================================

OH        = 8000.0          # operating hours/yr
C_furnace = 0.006           # USD/MJ  (natural gas)
C_elec    = 0.014           # USD/MJ  (electricity)
C_steam   = 0.003           # USD/MJ  (LP steam)
C_benz    = 1000.0          # USD/kmol benzene

P_sys     = 25.0 * 101325   # Pa  system pressure
P_FL_fixed = 25.0 * 101325  # Pa  flash pressure (isobaric)
P_flash_in = 20.0 * 101325  # Pa  flash inlet (slightly lower - gives PR=1.25)
PR        = P_sys / P_flash_in  # = 1.25  pressure ratio for compressor

kappa     = 1.4
eta_comp  = 0.85
R_gas     = 8.314

Cp = {'H2':29.1, 'CH4':35.7, 'Tol':103.7, 'Benz':82.4, 'Diph':165.0}

# T1 stabiliser split fractions (fraction to liquid bottoms)
sigma = {
    'H2':   0.02,
    'CH4':  0.05,
    'Tol':  0.98,
    'Benz': 0.95,
    'Diph': 0.999,
}

T_HX_cold_out = 400.0   # K  feed temperature after HX1 pre-heat


def build_hda():
    """Return (model, efmap, solver_kwargs) for the HDA case study, with the
    three black-box caches emptied so every solve starts cold."""
    # =============================================================================
    # Decision variable nominal values (initial guess)
    # =============================================================================

    F_H2_fresh_init  = 3.20
    F_Tol_fresh_init = 0.60
    phi_init         = 0.20
    V_R_init         = 3.0
    T_in_init        = 894.0
    T_FL_init        = 322.0
    RR2_init         = 3.0
    RR3_init         = 2.5


    # =============================================================================
    # Pre-solve: evaluate all three BBs at nominal point
    # Seeds all intermediate variable initialisations consistently
    # Eliminates non-zero linking constraint residuals at iteration 0
    # =============================================================================

    # Step 1: mixer composition at nominal (approximate gas recycle = 0 first pass)
    F_CH4_fresh_init = 0.03 * F_H2_fresh_init   # = 0.096 mol/s  [FIX 2]

    # First-pass mixer (no recycle yet - will be corrected after BB calls)
    F_H2_mix_init0   = F_H2_fresh_init
    F_CH4_mix_init0  = F_CH4_fresh_init
    F_Tol_mix_init0  = F_Tol_fresh_init
    F_Benz_mix_init0 = 0.001   # trace

    # Step 2: evaluate BB1 at nominal point
    _r = HDA_Reactor_sim(
        F_H2_mix_init0, F_CH4_mix_init0,
        F_Tol_mix_init0, F_Benz_mix_init0,
        T_in_init, V_R_init
    )
    (F_H2_Rout_init, F_CH4_Rout_init, F_Tol_Rout_init,
     F_Benz_Rout_init, F_Diph_Rout_init, T_Rout_init, H_Rout_init) = _r


    # Step 3: evaluate BB2 at nominal point
    _f = HDA_Flash_sim(
        F_H2_Rout_init, F_CH4_Rout_init, F_Tol_Rout_init,
        F_Benz_Rout_init, F_Diph_Rout_init,
        T_FL_init, P_FL_fixed
    )
    (F_H2_vap_init,  F_CH4_vap_init,  F_Tol_vap_init,
     F_Benz_vap_init, F_Diph_vap_init,
     F_H2_liq_init,  F_CH4_liq_init,  F_Tol_liq_init,
     F_Benz_liq_init, F_Diph_liq_init,
     H_vap_init, H_liq_init) = _f


    # Step 4: T1 stabiliser split
    F_Benz_T1_init = sigma['Benz'] * F_Benz_liq_init
    F_Tol_T1_init  = sigma['Tol']  * F_Tol_liq_init
    F_Diph_T1_init = sigma['Diph'] * F_Diph_liq_init

    # Step 5: evaluate BB3 at nominal point
    _d = HDA_Distillation_sim(
        F_Benz_T1_init, F_Tol_T1_init, F_Diph_T1_init,
        RR2_init, RR3_init
    )
    (F_Benz_prod_init, F_Tol_dist_init,
     F_Diph_out_init, Q_T2_init, Q_T3_init) = _d


    # Step 6: splitter recycles (consistent with phi_init)
    F_H2_rec_init   = (1.0 - phi_init) * F_H2_vap_init
    F_CH4_rec_init  = (1.0 - phi_init) * F_CH4_vap_init
    F_Benz_rec_init = (1.0 - phi_init) * F_Benz_vap_init
    F_Tol_rec_gas_init = (1.0 - phi_init) * F_Tol_vap_init  # [FIX 4]

    # Step 7: toluene recycle from distillation (closes loop)
    F_Tol_rec_init = F_Tol_dist_init   # [FIX 3 - replaces 0.55 hardcode]

    # Step 8: corrected mixer composition (with recycles)
    F_H2_mix_init   = F_H2_fresh_init  + F_H2_rec_init
    F_CH4_mix_init  = F_CH4_fresh_init + F_CH4_rec_init    # [FIX 2]
    F_Tol_mix_init  = F_Tol_fresh_init + F_Tol_rec_init + F_Tol_rec_gas_init
    F_Benz_mix_init = F_Benz_rec_init

    # Utility variables
    Q_H1_init = (
        F_H2_mix_init  * Cp['H2']  +
        F_CH4_mix_init * Cp['CH4'] +
        F_Tol_mix_init * Cp['Tol'] +
        F_Benz_mix_init* Cp['Benz']
    ) * (T_in_init - T_HX_cold_out) / 1e6   # MJ/s

    F_rec_total_init = F_H2_rec_init + F_CH4_rec_init + F_Benz_rec_init
    W_C_init = (
        F_rec_total_init * R_gas * T_FL_init / eta_comp *
        (kappa / (kappa - 1.0)) *
        (PR ** ((kappa - 1.0) / kappa) - 1.0)
    ) / 1e6   # MJ/s

    TAC_init = (
        C_furnace * Q_H1_init * 3600.0 * OH / 1e3 +
        C_elec    * W_C_init  * 3600.0 * OH / 1e3 +
        C_steam * (Q_T2_init + Q_T3_init) * 3600.0 * OH / 1e3 -
        C_benz * F_Benz_prod_init * 3.6 * OH / 1e3
    )



    # =============================================================================
    # Build Pyomo model
    # =============================================================================

    m = ConcreteModel()

    # ---------------------------------------------------------------------------
    # Fixed parameters (must NOT appear inside ExternalFunction calls)
    # ---------------------------------------------------------------------------
    m.P_sys   = Param(initialize=P_sys)
    m.kappa   = Param(initialize=kappa)
    m.eta_comp = Param(initialize=eta_comp)
    m.R_gas   = Param(initialize=R_gas)

    # ---------------------------------------------------------------------------
    # Decision variables  (bounds tightened per diagnostic report)
    # ---------------------------------------------------------------------------
    m.F_H2_fresh  = Var(bounds=(0.1,   8.0),   initialize=F_H2_fresh_init)  # lb raised: prevents near-zero reactor feed
    m.F_Tol_fresh = Var(bounds=(0.01,   3.0),   initialize=F_Tol_fresh_init)
    m.phi         = Var(bounds=(0.05,  0.45),  initialize=phi_init)         # lb raised: prevents purge collapse
    m.V_R         = Var(bounds=(1.5,   4.5),   initialize=V_R_init)         # ub reduced: prevents thermal runaway at high V_R
    m.T_in        = Var(bounds=(860.,  920.),  initialize=T_in_init)         # ub reduced: prevents T_out > 2000K runaway
    m.T_FL        = Var(bounds=(305.,  380.),  initialize=T_FL_init)
    m.RR2         = Var(bounds=(1.5,   10.0),  initialize=RR2_init)
    m.RR3         = Var(bounds=(1.5,   10.0),  initialize=RR3_init)

    # FIX 1: T_FL must also be a Var (already is) - confirm it's not passed as Param anywhere

    # ---------------------------------------------------------------------------
    # Mixer M1 outlet variables
    # ---------------------------------------------------------------------------
    m.F_H2_mix   = Var(bounds=(0.1,  20.0), initialize=F_H2_mix_init)
    m.F_CH4_mix  = Var(bounds=(0.0,   5.0), initialize=F_CH4_mix_init)
    m.F_Tol_mix  = Var(bounds=(0.1,   5.0), initialize=F_Tol_mix_init)
    m.F_Benz_mix = Var(bounds=(0.0,   2.0), initialize=F_Benz_mix_init)

    # Gas recycle streams (from splitter SP)
    m.F_H2_rec     = Var(bounds=(0.0, 25.0), initialize=F_H2_rec_init)
    m.F_CH4_rec    = Var(bounds=(0.0,  8.0), initialize=F_CH4_rec_init)
    m.F_Benz_rec   = Var(bounds=(0.0,  2.0), initialize=F_Benz_rec_init)
    m.F_Tol_rec_gas = Var(bounds=(0.0, 1.0), initialize=F_Tol_rec_gas_init)  # [FIX 4]

    # Toluene recycle from distillation
    m.F_Tol_rec    = Var(bounds=(0.0,  3.5), initialize=F_Tol_rec_init)    # [FIX bounds]

    # ---------------------------------------------------------------------------
    # BB1 reactor outlet variables
    # ---------------------------------------------------------------------------
    m.F_H2_Rout   = Var(bounds=(0.0,  20.0), initialize=F_H2_Rout_init)
    m.F_CH4_Rout  = Var(bounds=(0.0,  10.0), initialize=F_CH4_Rout_init)
    m.F_Tol_Rout  = Var(bounds=(0.0,   6.0), initialize=F_Tol_Rout_init)
    m.F_Benz_Rout = Var(bounds=(0.0,   6.0), initialize=F_Benz_Rout_init)
    m.F_Diph_Rout = Var(bounds=(0.0,   1.0), initialize=F_Diph_Rout_init)
    m.T_Rout      = Var(bounds=(800., 1200.), initialize=T_Rout_init)   # upper bound allows physically valid high-T but clips extreme runaway
    m.H_Rout      = Var(bounds=(-20.,  20.), initialize=H_Rout_init)       # [FIX bounds]

    # ---------------------------------------------------------------------------
    # BB2 flash outlet variables
    # ---------------------------------------------------------------------------
    m.F_H2_vap    = Var(bounds=(0.0,  25.0), initialize=F_H2_vap_init)    # [FIX bounds]
    m.F_CH4_vap   = Var(bounds=(0.0,  8.0), initialize=F_CH4_vap_init)
    m.F_Tol_vap   = Var(bounds=(0.0,   2.0), initialize=F_Tol_vap_init)
    m.F_Benz_vap  = Var(bounds=(0.0,   2.0), initialize=F_Benz_vap_init)
    m.F_Diph_vap  = Var(bounds=(0.0,   0.1), initialize=F_Diph_vap_init)
    m.F_H2_liq    = Var(bounds=(0.0,   2.0), initialize=F_H2_liq_init)
    m.F_CH4_liq   = Var(bounds=(0.0,   2.0), initialize=F_CH4_liq_init)
    m.F_Tol_liq   = Var(bounds=(0.0,   6.0), initialize=F_Tol_liq_init)
    m.F_Benz_liq  = Var(bounds=(0.0,   6.0), initialize=F_Benz_liq_init)
    m.F_Diph_liq  = Var(bounds=(0.0,   1.0), initialize=F_Diph_liq_init)
    m.H_vap       = Var(bounds=(-20.,  20.), initialize=H_vap_init)        # [FIX bounds]
    m.H_liq       = Var(bounds=(-20.,  20.), initialize=H_liq_init)        # [FIX bounds]

    # FIX 1: P_FL must be a Var (not Param) because it appears in ExternalFunction
    # calls. Fix bounds equal so it remains constant during optimisation.
    m.P_FL_var = Var(bounds=(P_FL_fixed, P_FL_fixed), initialize=P_FL_fixed)

    # T1 stabiliser bottoms
    m.F_Benz_T1   = Var(bounds=(0.0,   5.0), initialize=F_Benz_T1_init)
    m.F_Tol_T1    = Var(bounds=(0.0,   5.0), initialize=F_Tol_T1_init)
    m.F_Diph_T1   = Var(bounds=(0.0,   1.0), initialize=F_Diph_T1_init)

    # ---------------------------------------------------------------------------
    # BB3 distillation outlet variables
    # ---------------------------------------------------------------------------
    m.F_Benz_prod = Var(bounds=(0.0,   5.0),  initialize=F_Benz_prod_init)
    m.F_Tol_dist  = Var(bounds=(0.0,   3.0),  initialize=F_Tol_dist_init)
    m.F_Diph_out  = Var(bounds=(0.0,   1.0),  initialize=F_Diph_out_init)
    m.Q_T2        = Var(bounds=(0.0, 10.0),  initialize=Q_T2_init)
    m.Q_T3        = Var(bounds=(0.0,  5.0),  initialize=Q_T3_init)

    # ---------------------------------------------------------------------------
    # Utility variables
    # ---------------------------------------------------------------------------
    m.Q_H1  = Var(bounds=(1e-4, 50.0),  initialize=Q_H1_init)
    m.W_C   = Var(bounds=(1e-4, 20.0),  initialize=W_C_init)
    m.TAC   = Var(bounds=(-1e5,  1e5),  initialize=TAC_init)   # kUSD/yr [FIX 6]


    # =============================================================================
    # Glass-box constraints
    # =============================================================================

    # ---------------------------------------------------------------------------
    # M1 Mixer mole balances
    # FIX 4: F_Tol_mix includes gas-recycle toluene (F_Tol_rec_gas)
    # ---------------------------------------------------------------------------
    m.M1_H2  = Constraint(expr=
        m.F_H2_mix == m.F_H2_fresh + m.F_H2_rec)

    m.M1_CH4 = Constraint(expr=
        m.F_CH4_mix == 0.03 * m.F_H2_fresh + m.F_CH4_rec)

    m.M1_Tol = Constraint(expr=
        m.F_Tol_mix == m.F_Tol_fresh + m.F_Tol_rec + m.F_Tol_rec_gas)  # [FIX 4]

    m.M1_Benz = Constraint(expr=
        m.F_Benz_mix == m.F_Benz_rec)

    # ---------------------------------------------------------------------------
    # BB1 linking constraints
    # Inputs: F_H2_mix, F_CH4_mix, F_Tol_mix, F_Benz_mix, T_in, V_R  (all Var)
    # ---------------------------------------------------------------------------
    m.BB1_H2   = Constraint(expr=
        m.F_H2_Rout   == R_F_H2_out(  m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))
    m.BB1_CH4  = Constraint(expr=
        m.F_CH4_Rout  == R_F_CH4_out( m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))
    m.BB1_Tol  = Constraint(expr=
        m.F_Tol_Rout  == R_F_Tol_out( m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))
    m.BB1_Benz = Constraint(expr=
        m.F_Benz_Rout == R_F_Benz_out(m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))
    m.BB1_Diph = Constraint(expr=
        m.F_Diph_Rout == R_F_Diph_out(m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))
    m.BB1_T    = Constraint(expr=
        m.T_Rout      == R_T_out(     m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))
    m.BB1_H    = Constraint(expr=
        m.H_Rout      == R_H_out(     m.F_H2_mix, m.F_CH4_mix, m.F_Tol_mix, m.F_Benz_mix, m.T_in, m.V_R))

    # ---------------------------------------------------------------------------
    # Furnace H1: scaled energy balance [FIX 8]
    # Q_H1 [MJ/s] * 10 = sum(F_i * Cp_i) * (T_in - T_HX_cold_out) * 1e-5
    # Both sides O(1) after scaling
    # ---------------------------------------------------------------------------
    m.Furnace_c1 = Constraint(expr=
        m.Q_H1 * 10.0 == (
            m.F_H2_mix  * Cp['H2']   +
            m.F_CH4_mix * Cp['CH4']  +
            m.F_Tol_mix * Cp['Tol']  +
            m.F_Benz_mix* Cp['Benz']
        ) * (m.T_in - T_HX_cold_out) * 1e-5)

    # ---------------------------------------------------------------------------
    # BB2 linking constraints
    # FIX 1: m.P_FL_var (Var) used instead of m.P_FL_par (Param)
    # Inputs: reactor outlets (Var) + T_FL (Var) + P_FL_var (Var)
    # ---------------------------------------------------------------------------
    m.BB2_H2_vap   = Constraint(expr=
        m.F_H2_vap   == FL_F_H2_vap(  m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_CH4_vap  = Constraint(expr=
        m.F_CH4_vap  == FL_F_CH4_vap( m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Tol_vap  = Constraint(expr=
        m.F_Tol_vap  == FL_F_Tol_vap( m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Benz_vap = Constraint(expr=
        m.F_Benz_vap == FL_F_Benz_vap(m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Diph_vap = Constraint(expr=
        m.F_Diph_vap == FL_F_Diph_vap(m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_H2_liq   = Constraint(expr=
        m.F_H2_liq   == FL_F_H2_liq(  m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_CH4_liq  = Constraint(expr=
        m.F_CH4_liq  == FL_F_CH4_liq( m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Tol_liq  = Constraint(expr=
        m.F_Tol_liq  == FL_F_Tol_liq( m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Benz_liq = Constraint(expr=
        m.F_Benz_liq == FL_F_Benz_liq(m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Diph_liq = Constraint(expr=
        m.F_Diph_liq == FL_F_Diph_liq(m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Hvap     = Constraint(expr=
        m.H_vap      == FL_H_vap(     m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))
    m.BB2_Hliq     = Constraint(expr=
        m.H_liq      == FL_H_liq(     m.F_H2_Rout, m.F_CH4_Rout, m.F_Tol_Rout, m.F_Benz_Rout, m.F_Diph_Rout, m.T_FL, m.P_FL_var))

    # ---------------------------------------------------------------------------
    # SP Splitter: vapour -> purge + gas recycle
    # FIX 4: added m.SP_Tol for toluene in vapour stream
    # ---------------------------------------------------------------------------
    m.SP_H2   = Constraint(expr= m.F_H2_rec     == (1.0 - m.phi) * m.F_H2_vap)
    m.SP_CH4  = Constraint(expr= m.F_CH4_rec    == (1.0 - m.phi) * m.F_CH4_vap)
    m.SP_Benz = Constraint(expr= m.F_Benz_rec   == (1.0 - m.phi) * m.F_Benz_vap)
    m.SP_Tol  = Constraint(expr= m.F_Tol_rec_gas == (1.0 - m.phi) * m.F_Tol_vap)  # [FIX 4]

    # ---------------------------------------------------------------------------
    # C1 Compressor: isentropic work with physical PR=1.25 [FIX 7]
    # Scaled: both sides divided by 1e4 to bring to O(1-10)
    # ---------------------------------------------------------------------------
    m.Comp_c1 = Constraint(expr=
        m.W_C * 100.0 == (
            (m.F_H2_rec + m.F_CH4_rec + m.F_Benz_rec) *
            m.R_gas * m.T_FL / m.eta_comp
        ) * (m.kappa / (m.kappa - 1.0)) *
        (PR ** ((m.kappa - 1.0) / m.kappa) - 1.0) * 1e-4)

    # ---------------------------------------------------------------------------
    # T1 Stabiliser: fixed split fractions (glass-box)
    # ---------------------------------------------------------------------------
    m.T1_Benz = Constraint(expr= m.F_Benz_T1 == sigma['Benz'] * m.F_Benz_liq)
    m.T1_Tol  = Constraint(expr= m.F_Tol_T1  == sigma['Tol']  * m.F_Tol_liq)
    m.T1_Diph = Constraint(expr= m.F_Diph_T1 == sigma['Diph'] * m.F_Diph_liq)

    # ---------------------------------------------------------------------------
    # BB3 linking constraints
    # Inputs: T1 bottoms (Benz, Tol, Diph) + RR2 + RR3  (all Var)
    # ---------------------------------------------------------------------------
    m.BB3_Benz = Constraint(expr=
        m.F_Benz_prod == Dist_F_Benz_product(m.F_Benz_T1, m.F_Tol_T1, m.F_Diph_T1, m.RR2, m.RR3))
    m.BB3_Tol  = Constraint(expr=
        m.F_Tol_dist  == Dist_F_Tol_recycle( m.F_Benz_T1, m.F_Tol_T1, m.F_Diph_T1, m.RR2, m.RR3))
    m.BB3_Diph = Constraint(expr=
        m.F_Diph_out  == Dist_F_Diph_out(    m.F_Benz_T1, m.F_Tol_T1, m.F_Diph_T1, m.RR2, m.RR3))
    m.BB3_QT2  = Constraint(expr=
        m.Q_T2        == Dist_Q_T2(          m.F_Benz_T1, m.F_Tol_T1, m.F_Diph_T1, m.RR2, m.RR3))
    m.BB3_QT3  = Constraint(expr=
        m.Q_T3        == Dist_Q_T3(          m.F_Benz_T1, m.F_Tol_T1, m.F_Diph_T1, m.RR2, m.RR3))

    # ---------------------------------------------------------------------------
    # Recycle loop closure constraints
    # These are the key feasibility constraints the funnel must manage
    # ---------------------------------------------------------------------------
    # Toluene distillation recycle -> M1
    m.Tol_recycle_closure = Constraint(expr=
        m.F_Tol_rec == m.F_Tol_dist)

    # ---------------------------------------------------------------------------
    # Product and process specifications
    # ---------------------------------------------------------------------------
    m.Purity_spec = Constraint(expr=
        m.F_Benz_prod >= 0.997 * m.F_Benz_T1)

    m.Min_prod = Constraint(expr=
        m.F_Benz_prod >= 0.05)

    # ---------------------------------------------------------------------------
    # TAC definition: normalised so all Jacobian coefficients are O(1)
    # Divide through by _TAC_scale = C_benz * 3.6 * OH / 1e3 = 28800
    # This makes the F_Benz_prod coefficient exactly 1.0 in the Jacobian,
    # and all utility cost coefficients O(1e-4) to O(1e-2).
    # Without this normalisation the criticality check subproblem is ill-scaled
    # (coefficients span 8 orders of magnitude) causing IPOPT to declare infeasible.
    # ---------------------------------------------------------------------------
    _TAC_scale = C_benz * 3.6 * OH / 1e3   # = 28800 USD/(mol/s yr)

    m.TAC_def = Constraint(expr=
        m.TAC / _TAC_scale == (
            C_furnace * m.Q_H1 * 3600.0 * OH / 1e3 / _TAC_scale +
            C_elec    * m.W_C  * 3600.0 * OH / 1e3 / _TAC_scale +
            C_steam * (m.Q_T2 + m.Q_T3) * 3600.0 * OH / 1e3 / _TAC_scale -
            m.F_Benz_prod
        ))


    # =============================================================================
    # Objective: minimise TAC [kUSD/yr]
    # =============================================================================

    m.obj = Objective(expr=m.TAC, sense=minimize)


    # =============================================================================
    # efmap
    # =============================================================================

    efmap = {
        "block1": BB1_functions,    # 7  outputs: reactor ODE
        "block2": BB2_functions,    # 12 outputs: flash VLE
        "block3": BB3_functions,    # 5  outputs: distillation FUG
    }

    # Every run starts with empty black-box caches
    TRF_CACHE_R.clear()
    TRF_CACHE_FL.clear()
    TRF_CACHE_DIST.clear()

    return m, efmap, dict(scaling=0)
