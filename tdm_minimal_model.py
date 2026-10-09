from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

from tdm_minimal_io import (
    LATEST_RESULTS_CSV,
    LATEST_RESULTS_HTML,
    LATEST_RESULTS_MANIFEST,
    MODEL_EDM_CPA,
    MODEL_EDM_LANGMUIR,
    MODEL_LABELS,
    MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR,
    MODEL_TDM_WANG,
    MAX_HIC_LOG_AFFINITY,
    RESULT_GUARD_BUILD,
    active_components,
    apply_batch_shared_parameters,
    batch_parameter_fingerprint,
    batch_shared_parameter_snapshot,
    buffer_blend_chemistry,
    buffer_chemistry,
    derived_input_summary,
    required_parameter_values,
    configuration_fingerprint,
    normalize_config,
    normalize_load_amount_basis,
    process_pH_span,
    resolve_process_chemistry,
    salt_input_mode,
    uses_cpa,
    uses_edm,
    uses_langmuir,
    uses_tdm,
    is_hic,
    load_material_chemistry_is_explicit,
    validate_config,
)

from tdm_mobile_phase import MobilePhaseTransport, delay_by_volume

# Physical constants and fixed values stated in the supplied CPA paper.
E_CHARGE = 1.602176634e-19
N_A = 6.02214076e23
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
CPA_TEMPERATURE_K = 298.15
CPA_RELATIVE_PERMITTIVITY = 78.3
CPA_CEX_LIGAND_PK = 2.3
SMALL = 1e-30
MAX_EXP = MAX_HIC_LOG_AFFINITY

# Briskot et al. Eq. 23; inverted only for mass/molar conversion when the user
# supplies diameter as requested by the Word document.
CPA_RADIUS_FROM_MW_A_NM = 5.1
CPA_RADIUS_FROM_MW_B_NM = -5.4


@dataclass
class ProgramStage:
    name: str
    kind: str
    duration_CV: float
    flow_mL_min: float
    mode: str
    start_pH: float
    end_pH: float
    start_I_M: float
    end_I_M: float
    start_salt_M: float
    end_salt_M: float
    start_conductivity_mS_cm: float
    end_conductivity_mS_cm: float
    feed_vector: np.ndarray
    start_percent_B: float | None = None
    end_percent_B: float | None = None

    def environment(self, frac: float) -> tuple[float, float]:
        frac = float(np.clip(frac, 0.0, 1.0))
        mode = str(self.mode or "SET_POINT").upper()
        if mode == "LINEAR":
            f = frac
        elif mode == "STEP":
            f = 1.0
        else:
            f = 0.0
        pH = self.start_pH + f * (self.end_pH - self.start_pH)
        ionic = self.start_I_M + f * (self.end_I_M - self.start_I_M)
        return float(pH), float(max(ionic, 0.0))

    def chemistry(self, frac: float) -> tuple[float, float, float, float]:
        frac = float(np.clip(frac, 0.0, 1.0))
        mode = str(self.mode or "SET_POINT").upper()
        if mode == "LINEAR": f = frac
        elif mode == "STEP": f = 1.0
        else: f = 0.0
        pH = self.start_pH + f * (self.end_pH - self.start_pH)
        ionic = self.start_I_M + f * (self.end_I_M - self.start_I_M)
        salt = self.start_salt_M + f * (self.end_salt_M - self.start_salt_M)
        cond = self.start_conductivity_mS_cm + f * (self.end_conductivity_mS_cm - self.start_conductivity_mS_cm)
        return float(pH), float(max(salt, 0.0)), float(max(ionic, 0.0)), float(max(cond, 0.0))


@dataclass
class StageProgram:
    column_volume_mL: float
    stages: list[ProgramStage]

    def __post_init__(self) -> None:
        self.stage_start_time_s: list[float] = []
        self.stage_end_time_s: list[float] = []
        self.stage_start_CV: list[float] = []
        self.stage_end_CV: list[float] = []
        t = 0.0
        cv = 0.0
        for stage in self.stages:
            self.stage_start_time_s.append(t)
            self.stage_start_CV.append(cv)
            dt = stage.duration_CV * self.column_volume_mL / stage.flow_mL_min * 60.0
            t += dt
            cv += stage.duration_CV
            self.stage_end_time_s.append(t)
            self.stage_end_CV.append(cv)
        self.total_time_s = t
        self.total_CV = cv
        self.b_load = self.stage_end_CV[0] if self.stages else 0.0
        plw_end = self.b_load
        for idx, stage in enumerate(self.stages):
            if stage.kind == "PLW":
                plw_end = self.stage_end_CV[idx]
        self.b_wash = plw_end
        self.b_elution = self.total_CV
        self.cv_time_s = min(
            (self.column_volume_mL / max(stage.flow_mL_min, SMALL) * 60.0 for stage in self.stages),
            default=60.0,
        )
        first = self.stages[0] if self.stages else ProgramStage("Load", "LOAD", 1, 1, "SET_POINT", 7, 7, 0, 0, 0, 0, 0, 0, np.array([]))
        self.pH_A = first.start_pH
        self.ionic_A_M = first.start_I_M

    def _stage_index_time(self, t_s: float) -> int:
        if not self.stages:
            return 0
        t = max(float(t_s), 0.0)
        for i, end in enumerate(self.stage_end_time_s):
            if t < end - 1e-12:
                return i
        return len(self.stages) - 1

    def _stage_index_cv(self, cv: float) -> int:
        if not self.stages:
            return 0
        v = max(float(cv), 0.0)
        for i, end in enumerate(self.stage_end_CV):
            if v < end - 1e-12:
                return i
        return len(self.stages) - 1

    def time_to_cv(self, t_s: float) -> float:
        if not self.stages:
            return 0.0
        t = float(np.clip(t_s, 0.0, self.total_time_s))
        i = self._stage_index_time(t)
        stage = self.stages[i]
        local_t = t - self.stage_start_time_s[i]
        cv = self.stage_start_CV[i] + local_t * stage.flow_mL_min / (self.column_volume_mL * 60.0)
        return float(np.clip(cv, 0.0, self.total_CV))

    def cv_to_time(self, cv: float) -> float:
        if not self.stages:
            return 0.0
        v = float(np.clip(cv, 0.0, self.total_CV))
        i = self._stage_index_cv(v)
        stage = self.stages[i]
        local_cv = v - self.stage_start_CV[i]
        return self.stage_start_time_s[i] + local_cv * self.column_volume_mL / stage.flow_mL_min * 60.0

    def flow_at_time(self, t_s: float) -> float:
        return float(self.stages[self._stage_index_time(t_s)].flow_mL_min)

    def flow_at_cv(self, cv: float) -> float:
        return float(self.stages[self._stage_index_cv(cv)].flow_mL_min)

    def at_time(self, t_s: float) -> tuple[np.ndarray, float, float, float]:
        i = self._stage_index_time(t_s)
        stage = self.stages[i]
        start = self.stage_start_time_s[i]
        end = self.stage_end_time_s[i]
        frac = 1.0 if end <= start else (float(t_s) - start) / (end - start)
        pH, ionic = stage.environment(frac)
        protein = stage.feed_vector.copy() if stage.kind == "LOAD" else np.zeros_like(stage.feed_vector)
        return protein, pH, ionic, float(i + 1)

    def at_cv(self, cv: float) -> tuple[np.ndarray, float, float, float]:
        i = self._stage_index_cv(cv)
        stage = self.stages[i]
        start = self.stage_start_CV[i]
        end = self.stage_end_CV[i]
        frac = 1.0 if end <= start else (float(cv) - start) / (end - start)
        pH, ionic = stage.environment(frac)
        protein = stage.feed_vector.copy() if stage.kind == "LOAD" else np.zeros_like(stage.feed_vector)
        return protein, pH, ionic, float(i + 1)

    def chemistry_at_cv(self, cv: float) -> tuple[float, float, float, float, float]:
        i = self._stage_index_cv(cv)
        stage = self.stages[i]
        start = self.stage_start_CV[i]
        end = self.stage_end_CV[i]
        frac = 1.0 if end <= start else (float(cv) - start) / (end - start)
        pH, salt, ionic, cond = stage.chemistry(frac)
        return pH, salt, ionic, cond, float(i + 1)

    def percent_B_at_cv(self, cv: float) -> float:
        """Return the programmed outlet %B, or NaN when a stage is not %B-controlled."""
        i = self._stage_index_cv(cv)
        stage = self.stages[i]
        if stage.start_percent_B is None or stage.end_percent_B is None:
            return float("nan")
        start = self.stage_start_CV[i]
        end = self.stage_end_CV[i]
        frac = 1.0 if end <= start else (float(cv) - start) / (end - start)
        frac = float(np.clip(frac, 0.0, 1.0))
        mode = str(stage.mode or "SET_POINT").upper()
        if mode == "LINEAR":
            f = frac
        elif mode == "STEP":
            f = 1.0
        else:
            f = 0.0
        return float(stage.start_percent_B + f * (stage.end_percent_B - stage.start_percent_B))

    def environment_with_derivative(self, t_s: float) -> tuple[float, float, float, float, float]:
        i = self._stage_index_time(t_s)
        stage = self.stages[i]
        start = self.stage_start_time_s[i]
        end = self.stage_end_time_s[i]
        duration = max(end - start, SMALL)
        frac = (float(t_s) - start) / duration
        pH, ionic = stage.environment(frac)
        if str(stage.mode).upper() == "LINEAR":
            dpH_dt = (stage.end_pH - stage.start_pH) / duration
            dI_dt = (stage.end_I_M - stage.start_I_M) / duration
        else:
            dpH_dt = dI_dt = 0.0
        return pH, ionic, float(i + 1), float(dpH_dt), float(dI_dt)

    @property
    def stage_boundaries_s(self) -> list[float]:
        return list(self.stage_end_time_s)

    def flow_array_L_s(self, times: np.ndarray) -> np.ndarray:
        return np.asarray([self.flow_at_time(float(t)) / 1000.0 / 60.0 for t in times], dtype=float)


@dataclass
class CPAComponent:
    name: str
    mass_fraction: float
    radius_m: float
    mw_kDa: float
    As_m_inv: float
    Z_ref: float
    pH_ref: float
    Z_poly: tuple[float, float, float]
    delta_ref: float
    delta_pH_slope_m2_C: float
    # The 2025 calibration paper neglected Z_lat,i; keep it fixed internally.
    salt_sensitivity_per_M: float = 0.0
    Z_lat: float = 0.0
    k_eff_m_s: float = 0.0
    kkin_star_s: float = 0.0

    def Z(self, pH: float) -> float:
        delta = float(pH) - self.pH_ref
        return self.Z_ref + self.Z_poly[0] * delta + self.Z_poly[1] * delta**2 + self.Z_poly[2] * delta**3

    def sigma(self, pH: float) -> float:
        return E_CHARGE * self.Z(pH) / (4.0 * math.pi * self.radius_m**2)

    def delta(self, pH: float) -> float:
        """CPA interaction/boundary-layer parameter Delta_i.

        For constant-pH operation this is simply delta_ref.  When EDM+CPA is
        deliberately run across a pH range, the older Briskot empirical
        relationship is applied to log10(Delta_i) using the charge-density
        change.
        """
        if abs(self.delta_pH_slope_m2_C) <= 0.0:
            return self.delta_ref
        sigma = abs(self.sigma(pH))
        sigma_ref = abs(E_CHARGE * self.Z_ref / (4.0 * math.pi * self.radius_m**2))
        log10_delta = math.log10(max(self.delta_ref, 1e-300)) + self.delta_pH_slope_m2_C * (sigma - sigma_ref)
        return 10.0 ** float(np.clip(log10_delta, -300.0, 300.0))


class CPAModel:
    """CPA equilibrium and kinetic terms used by both EDM and TDM.

    The supplied CPA paper refers detailed expressions for the protein/adsorber
    electrostatic energy and available-surface term to earlier derivations. Those
    sub-expressions are retained from the uploaded core model. Z_lat,i is fixed to
    zero because the 2025 calibration paper explicitly neglected it; it is not a
    user-facing parameter.
    """

    def __init__(self, components: list[CPAComponent], ligand_density_mol_m2: float, system_scale: float, hic: bool = False):
        self.hic = hic
        self.components = components
        self.n = len(components)
        self.ligand_density_mol_m2 = float(ligand_density_mol_m2)
        self.system_scale = float(system_scale)
        self.eps = CPA_RELATIVE_PERMITTIVITY * EPS0
        self.kBT = K_B * CPA_TEMPERATURE_K
        self.has_lateral_interactions = any(abs(c.Z_lat) > 0.0 for c in components)

    def ligand_ionisation(self, pH: float) -> float:
        exponent = float(np.clip(CPA_CEX_LIGAND_PK - pH, -300.0, 300.0))
        return 1.0 / (1.0 + 10.0**exponent)

    def sigma_adsorber(self, pH: float) -> float:
        return -E_CHARGE * N_A * self.ligand_density_mol_m2 * self.ligand_ionisation(pH)

    def debye_kappa(self, ionic_strength_mol_m3: float) -> float:
        ionic = max(float(ionic_strength_mol_m3), 1e-12)
        return math.sqrt(2.0 * E_CHARGE**2 * N_A * ionic / (self.eps * K_B * CPA_TEMPERATURE_K))

    def surface_potential_linear(self, sigma: float, kappa: float) -> float:
        return sigma / max(self.eps * kappa, SMALL)

    def u_adsorber(self, comp: CPAComponent, pH: float, ionic_strength_mol_m3: float) -> float:
        kappa = self.debye_kappa(ionic_strength_mol_m3)
        psi_a = self.surface_potential_linear(self.sigma_adsorber(pH), kappa)
        psi_p = self.surface_potential_linear(comp.sigma(pH), kappa)
        separation = max(0.15e-9, 0.02 * comp.radius_m)
        return self.system_scale * 4.0 * math.pi * self.eps * comp.radius_m * psi_a * psi_p * math.exp(-kappa * separation)

    def surface_coverages(self, qv: np.ndarray) -> tuple[float, np.ndarray]:
        qv = np.maximum(np.asarray(qv, dtype=float), 0.0)
        q_surface = np.array([qv[i] / max(self.components[i].As_m_inv, SMALL) for i in range(self.n)])
        theta = sum(math.pi * N_A * self.components[i].radius_m**2 * q_surface[i] for i in range(self.n))
        return max(float(theta), 0.0), q_surface

    def available_surface(self, i: int, qv: np.ndarray) -> float:
        theta, q_surface = self.surface_coverages(qv)
        theta = min(max(theta, 0.0), 0.999999)
        r = self.components[i].radius_m
        surface_density = N_A * float(np.sum(q_surface))
        radius_density = N_A * float(sum(self.components[j].radius_m * q_surface[j] for j in range(self.n)))
        one_minus = max(1.0 - theta, SMALL)
        work = (
            (math.pi * r*r * surface_density + 2.0 * math.pi * r * radius_density) / one_minus
            + math.pi * r*r * radius_density*radius_density / (one_minus*one_minus)
        )
        return float(np.clip(one_minus * math.exp(-min(max(work, 0.0), MAX_EXP)), 0.0, 1.0))

    def lateral_interactions(self, qv: np.ndarray, ionic_strength_mol_m3: float) -> np.ndarray:
        if not self.has_lateral_interactions:
            return np.zeros(self.n)
        _, q_surface = self.surface_coverages(qv)
        number_density = max(N_A * float(np.sum(q_surface)), 0.0)
        if number_density <= 0:
            return np.zeros(self.n)
        kappa = self.debye_kappa(ionic_strength_mol_m3)
        hex_distance = math.sqrt(2.0 / (math.sqrt(3.0) * number_density))
        x = float(np.clip(kappa * hex_distance, 1e-8, 80.0))
        lattice_factor = math.sqrt(3.0) * hex_distance * N_A * math.exp(-x) / max(1.0 - math.exp(-x), SMALL)
        out = np.zeros(self.n)
        for i, ci in enumerate(self.components):
            total = 0.0
            for j, cj in enumerate(self.components):
                exp_arg = float(np.clip(kappa * (ci.radius_m + cj.radius_m), -80.0, 80.0))
                denom = max((1.0 + kappa * ci.radius_m) * (1.0 + kappa * cj.radius_m), SMALL)
                alpha = E_CHARGE**2 * ci.Z_lat * cj.Z_lat * math.exp(exp_arg) / (4.0 * math.pi * self.eps * denom)
                total += q_surface[j] * alpha
            out[i] = max(0.0, lattice_factor * total)
        return out

    @staticmethod
    def _fkin(y: float) -> float:
        ay = abs(y)
        if ay < 1e-5:
            return 1.0 - y*y/12.0
        if ay > 50.0:
            return y*y * math.exp(-ay)
        return 0.5 * y*y / max(math.cosh(y) - 1.0, SMALL)

    def equilibrium_factors(self, qv: np.ndarray, pH: float, ionic_strength_M: float) -> np.ndarray:
        if self.hic:
            # Phenomenological HIC extension of the CPA available-surface model.
            # The attached CPA paper is IEX: its electrostatic salt screening
            # must not be used to describe hydrophobic binding.
            return np.asarray([
                comp.delta_ref * self.available_surface(i, qv)
                * math.exp(float(np.clip(comp.salt_sensitivity_per_M * max(ionic_strength_M, 0.0), 0.0, MAX_HIC_LOG_AFFINITY)))
                for i, comp in enumerate(self.components)
            ])
        ionic_mol_m3 = max(float(ionic_strength_M), 0.0) * 1000.0
        lateral = self.lateral_interactions(qv, ionic_mol_m3)
        K = np.zeros(self.n)
        for i, comp in enumerate(self.components):
            y = float(np.clip(self.u_adsorber(comp, pH, ionic_mol_m3) / self.kBT, -80.0, 80.0))
            available = self.available_surface(i, qv)
            integral = 1.0 if abs(y) < 1e-7 else (1.0 - math.exp(float(np.clip(-y, -80.0, 80.0)))) / y
            K[i] = max(
                comp.delta(pH)
                * available
                * math.exp(float(np.clip(-lateral[i] / self.kBT, -80.0, 80.0)))
                * integral,
                0.0,
            )
        return K

    def kinetic_rate(self, cp: np.ndarray, qv: np.ndarray, pH: float, ionic_strength_M: float) -> np.ndarray:
        K = self.equilibrium_factors(qv, pH, ionic_strength_M)
        ionic_mol_m3 = max(float(ionic_strength_M), 0.0) * 1000.0
        dq = np.zeros(self.n)
        for i, comp in enumerate(self.components):
            y = 0.0 if self.hic else float(np.clip(self.u_adsorber(comp, pH, ionic_mol_m3) / self.kBT, -80.0, 80.0))
            kkin = max(comp.kkin_star_s * self._fkin(y), 0.0)
            dq[i] = kkin * (K[i] * max(float(cp[i]), 0.0) - max(float(qv[i]), 0.0))
        return dq

    def equilibrium_from_total(
        self, total: np.ndarray, phase_ratio_F: float, pH: float, ionic_strength_M: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert total = c + F q together with CPA equilibrium q=K(q)c.

        This conservative inversion is used by EDM+CPA. It requires no kinetic
        rate or extra numerical/user parameter.
        """
        total = np.maximum(np.asarray(total, dtype=float), 0.0)
        Fphase = max(float(phase_ratio_F), 1e-15)
        if not np.any(total > 0):
            return np.zeros(self.n), np.zeros(self.n)

        upper = total / Fphase
        if self.n == 1:
            mt = float(total[0])
            hi = float(upper[0])
            if hi <= 0:
                return np.array([mt]), np.zeros(1)
            def fscalar(qs: float) -> float:
                qarr = np.array([max(qs, 0.0)])
                c = max(mt - Fphase * qarr[0], 0.0)
                K = float(self.equilibrium_factors(qarr, pH, ionic_strength_M)[0])
                return float(qarr[0] - K * c)
            qv = brentq(fscalar, 0.0, hi, xtol=1e-12, rtol=1e-10, maxiter=60)
            q = np.array([qv])
            c = np.maximum(total - Fphase*q, 0.0)
            return c, q

        k0 = self.equilibrium_factors(np.zeros(self.n), pH, ionic_strength_M)
        q = np.minimum(k0 * total / np.maximum(1.0 + Fphase*k0, SMALL), upper)
        q = np.maximum(q, 0.0)

        def residual(qv: np.ndarray) -> np.ndarray:
            qv = np.clip(qv, 0.0, upper)
            c = np.maximum(total - Fphase*qv, 0.0)
            return qv - self.equilibrium_factors(qv, pH, ionic_strength_M) * c

        for _ in range(30):
            F0 = residual(q)
            scale = max(1.0, float(np.max(np.abs(q))))
            if float(np.max(np.abs(F0))) <= 2e-9 * scale:
                break
            A = np.eye(self.n)
            for j in range(self.n):
                h = max(1e-10, abs(float(q[j])) * 2e-5, float(upper[j]) * 1e-7)
                qp = q.copy(); qp[j] = min(qp[j] + h, upper[j])
                actual = qp[j] - q[j]
                if actual <= 0:
                    continue
                A[:, j] = (residual(qp) - F0) / actual
            try:
                step = np.linalg.solve(A, F0)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A, F0, rcond=None)[0]
            base = float(np.linalg.norm(F0))
            lam = 1.0
            accepted = False
            for _ in range(8):
                trial = np.clip(q - lam*step, 0.0, upper)
                if float(np.linalg.norm(residual(trial))) < base:
                    q = trial; accepted = True; break
                lam *= 0.5
            if not accepted:
                c = np.maximum(total - Fphase*q, 0.0)
                target = self.equilibrium_factors(q, pH, ionic_strength_M) * c
                q = np.clip(0.7*q + 0.3*target, 0.0, upper)
        c = np.maximum(total - Fphase*q, 0.0)
        return c, q


def _mw_from_radius_nm(radius_nm: float) -> float:
    return 10.0 ** ((float(radius_nm) - CPA_RADIUS_FROM_MW_B_NM) / CPA_RADIUS_FROM_MW_A_NM)


def _scalar_transport(field: np.ndarray, inlet: float, u: float, D: float, dx: float) -> np.ndarray:
    """Conservative finite-volume transport with limited second-order advection.

    First-order upwinding added substantial artificial axial dispersion on the
    coarse grids used by the JMP interface.  The monotonized-central limiter
    reconstructs a bounded face value in smooth regions and falls back to
    first order near extrema, while keeping the flux difference conservative.
    Molecular/axial dispersion remains centered at each interior face.
    """
    field = np.asarray(field, dtype=float)
    n = len(field)
    slope = np.zeros(n, dtype=float)
    if n > 2:
        left = field[1:-1] - field[:-2]
        right = field[2:] - field[1:-1]
        centered = 0.5 * (field[2:] - field[:-2])
        same_sign = left * right > 0.0
        magnitude = np.minimum(np.minimum(2.0 * np.abs(left), np.abs(centered)), 2.0 * np.abs(right))
        slope[1:-1] = np.where(same_sign, np.sign(centered) * magnitude, 0.0)

    flux = np.empty(n + 1, dtype=float)
    flux[0] = u * inlet
    if n > 1:
        if u >= 0:
            advective_face = field[:-1] + 0.5 * slope[:-1]
        else:
            advective_face = field[1:] - 0.5 * slope[1:]
        flux[1:-1] = u * advective_face - D * np.diff(field) / dx
    flux[-1] = u * field[-1]
    return -np.diff(flux) / dx


def _build_program(config: dict[str, Any], feed_vector: np.ndarray) -> StageProgram:
    model = config["model"]
    feed = config["feed"]
    process = config["process"]
    Vcol = float(config["column"]["volume_mL"])
    load_flow = float(config["column"]["flow_mL_min"])
    if normalize_load_amount_basis(process.get("load_amount_basis")) == "CAPACITY":
        load_cv = float(feed["load_density_mg_mL_resin"]) / float(feed["total_concentration_mg_mL"])
    else:
        load_cv = float(process.get("load_CV") or (
            float(feed["load_density_mg_mL_resin"]) / float(feed["total_concentration_mg_mL"])
        ))

    # Resolve the run chemistry for every model combination so the exact same
    # Buffer A/B and %B program is carried through every saved run. CPA models
    # consume pH/ionic strength in the adsorption calculation. HIC Competitive
    # Langmuir consumes local salt through its exponential affinity term; pH
    # remains a reported process condition for that isotherm.
    load_control = str(process.get("load_chemistry_control") or "BUFFER_B_PERCENT").strip().upper()
    if not load_material_chemistry_is_explicit(process):
        load_start_B = float(process.get("load_start_percent_B", 0.0))
        load_end_B = float(process.get("load_end_percent_B", load_start_B))
        load_start_env = buffer_blend_chemistry(config, load_start_B)
        load_end_env = buffer_blend_chemistry(config, load_end_B)
    else:
        # Explicit chemistry mode preserves older Buffer A/Buffer B/direct
        # recipes and uses the explicitly selected salt input.
        load_start_env = resolve_process_chemistry(
            config, source=process.get("load_source"), pH=process.get("load_pH"),
            salt_M=process.get("load_salt_concentration_M"),
            conductivity_mS_cm=process.get("load_conductivity_mS_cm"),
            salt_input_mode=process.get("load_salt_input_mode"),
        )
        load_end_env = load_start_env
        load_start_B = load_end_B = None
    load_pH = float(load_start_env["pH"] if load_start_env["pH"] is not None else 7.0)
    load_end_pH = float(load_end_env["pH"] if load_end_env["pH"] is not None else load_pH)
    load_salt = float(load_start_env["salt_concentration_M"] if load_start_env["salt_concentration_M"] is not None else 0.0)
    load_end_salt = float(load_end_env["salt_concentration_M"] if load_end_env["salt_concentration_M"] is not None else load_salt)
    load_I = float(load_start_env["ionic_strength_M"] if load_start_env["ionic_strength_M"] is not None else load_salt)
    load_end_I = float(load_end_env["ionic_strength_M"] if load_end_env["ionic_strength_M"] is not None else load_end_salt)
    load_cond = float(load_start_env["conductivity_mS_cm"] if load_start_env["conductivity_mS_cm"] is not None else 0.0)
    load_end_cond = float(load_end_env["conductivity_mS_cm"] if load_end_env["conductivity_mS_cm"] is not None else load_cond)

    stages: list[ProgramStage] = [ProgramStage(
        name="Load", kind="LOAD", duration_CV=load_cv, flow_mL_min=load_flow,
        mode=str(process.get("load_mode") or "LINEAR").upper(),
        start_pH=load_pH, end_pH=load_end_pH, start_I_M=load_I, end_I_M=load_end_I,
        start_salt_M=load_salt, end_salt_M=load_end_salt,
        start_conductivity_mS_cm=load_cond, end_conductivity_mS_cm=load_end_cond,
        feed_vector=np.asarray(feed_vector, dtype=float),
        start_percent_B=load_start_B, end_percent_B=load_end_B,
    )]
    for kind, key, count_key in (("PLW", "plw_steps", "plw_count"), ("ELUTION", "elution_steps", "elution_count")):
        count = int(float(process.get(count_key, 0) or 0))
        for i, raw in enumerate(list(process.get(key, []))[:count], start=1):
            mode = str(raw.get("mode") or "SET_POINT").upper()
            control = str(raw.get("chemistry_control") or "BUFFER_B_PERCENT").strip().upper()
            start_B = None
            end_B = None
            if control == "BUFFER_B_PERCENT":
                start_B = raw.get("start_percent_B")
                end_B = raw.get("end_percent_B")
                if start_B is None:
                    start_B = end_B
                if end_B is None:
                    end_B = start_B
                start_env = buffer_blend_chemistry(config, start_B)
                end_env = buffer_blend_chemistry(config, end_B)
            else:
                start_env = resolve_process_chemistry(
                    config, source=raw.get("start_source"), pH=raw.get("start_pH"),
                    salt_M=raw.get("start_salt_concentration_M"),
                    conductivity_mS_cm=raw.get("start_conductivity_mS_cm"),
                    salt_input_mode=raw.get("start_salt_input_mode"),
                )
                end_env = resolve_process_chemistry(
                    config, source=raw.get("end_source"), pH=raw.get("end_pH"),
                    salt_M=raw.get("end_salt_concentration_M"),
                    conductivity_mS_cm=raw.get("end_conductivity_mS_cm"),
                    salt_input_mode=raw.get("end_salt_input_mode"),
                )
            start_pH = float(start_env["pH"] if start_env["pH"] is not None else load_pH)
            end_pH = float(end_env["pH"] if end_env["pH"] is not None else start_pH)
            start_salt = float(start_env["salt_concentration_M"] if start_env["salt_concentration_M"] is not None else load_salt)
            end_salt = float(end_env["salt_concentration_M"] if end_env["salt_concentration_M"] is not None else start_salt)
            start_I = float(start_env["ionic_strength_M"] if start_env["ionic_strength_M"] is not None else start_salt)
            end_I = float(end_env["ionic_strength_M"] if end_env["ionic_strength_M"] is not None else end_salt)
            start_cond = float(start_env["conductivity_mS_cm"] if start_env["conductivity_mS_cm"] is not None else 0.0)
            end_cond = float(end_env["conductivity_mS_cm"] if end_env["conductivity_mS_cm"] is not None else start_cond)
            display_kind = "PLW" if kind == "PLW" else "Elution"
            stages.append(ProgramStage(
                name=f"{display_kind} {i}", kind=kind, duration_CV=float(raw["CV"]),
                flow_mL_min=float(raw["flow_mL_min"]), mode=mode,
                start_pH=start_pH, end_pH=end_pH, start_I_M=start_I, end_I_M=end_I,
                start_salt_M=start_salt, end_salt_M=end_salt,
                start_conductivity_mS_cm=start_cond, end_conductivity_mS_cm=end_cond,
                feed_vector=np.asarray(feed_vector, dtype=float),
                start_percent_B=(float(start_B) if control == "BUFFER_B_PERCENT" and start_B is not None else None),
                end_percent_B=(float(end_B) if control == "BUFFER_B_PERCENT" and end_B is not None else None),
            ))
    return StageProgram(column_volume_mL=Vcol, stages=stages)


def _langmuir_arrays(config: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    config = normalize_config(config)
    rows = active_components(config)
    qmax = np.array([float(r["qmax_g_L"]) for r in rows], dtype=float)
    b = np.array([float(r["b_L_g"]) for r in rows], dtype=float)
    # c and qmax use g/L, respectively of mobile and non-mobile stationary
    # phase. Legacy qmax in mg/mL is normalized with factor 1 before this point.
    H = qmax * b
    return rows, H, b


def _langmuir_q_jac(
    c: np.ndarray, H: np.ndarray, b: np.ndarray, *, salt_M: float = 0.0,
    salt_sensitivity_per_M: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Competitive HIC Langmuir loading and Jacobian at local salt concentration.

    The exponentially modified Langmuir form is q_i = qmax_i*b_i*exp(k_s,i*m)*c_i /
    (1 + sum_j b_j*exp(k_s,j*m)*c_j). The empirical k_s is calibrated to
    analytical salt molarity m [M] in the current run recipe.
    """
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    k_s = np.zeros_like(b) if salt_sensitivity_per_M is None else salt_sensitivity_per_M
    b_eff, H_eff = _langmuir_salt_affinity(H, b, k_s, salt_M)
    den = 1.0 + float(np.dot(b_eff, c))
    q = H_eff * c / max(den, SMALL)
    n = len(c)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = (H_eff[i] / den if i == j else 0.0) - H_eff[i] * c[i] * b_eff[j] / (den*den)
    return q, J


def _langmuir_salt_affinity(
    H: np.ndarray, b: np.ndarray, k_s: np.ndarray, salt_M: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the paper's H_i=qmax_i*b_i relation at each local salt value."""
    salt = np.maximum(np.asarray(salt_M, dtype=float), 0.0)
    k_s = np.asarray(k_s, dtype=float)
    factor = np.exp(np.clip(salt[..., None] * k_s, -MAX_HIC_LOG_AFFINITY, MAX_HIC_LOG_AFFINITY))
    return np.asarray(b, dtype=float) * factor, np.asarray(H, dtype=float) * factor


def _langmuir_c_q_from_total(
    total: np.ndarray, H: np.ndarray, b: np.ndarray, *, salt_M: float | np.ndarray,
    salt_sensitivity_per_M: np.ndarray, phase_ratio: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert total_i = c_i + F_i*q_i for salt-dependent competitive Langmuir.

    This keeps the conserved protein state independent of salt while allowing
    salt to change the instantaneous equilibrium split between mobile and bound
    protein. The shared competitive denominator reduces the multicomponent
    inversion to one monotone scalar root per axial cell.
    """
    total = np.maximum(np.asarray(total, dtype=float), 0.0)
    b = np.asarray(b, dtype=float)
    H = np.asarray(H, dtype=float)
    F = np.broadcast_to(np.asarray(phase_ratio, dtype=float), total.shape)
    b_eff, H_eff = _langmuir_salt_affinity(H, b, salt_sensitivity_per_M, salt_M)
    if not np.any(total > 0.0):
        return np.zeros_like(total), np.zeros_like(total)

    weighted_total = b_eff * total
    phase_capacity = F * H_eff
    # D=1+sum(b_eff*c) can span many orders of magnitude at high HIC salt.
    # Solve the bounded, strictly decreasing equation
    #   1/D + sum(b_eff*total/(D+F*H_eff)) - 1 = 0
    # in log(D). Unlike a residual scaled by D, this controls the free-protein
    # split when D is large and keeps the paper's shared denominator intact.
    lower = np.zeros(total.shape[:-1], dtype=float)
    upper = np.log1p(np.sum(weighted_total, axis=-1))
    log_den = 0.5 * (lower + upper)
    for _ in range(80):
        den = np.exp(log_den)
        den_i = den[..., None] + phase_capacity
        occupancy = weighted_total / den_i
        residual = 1.0 / den + np.sum(occupancy, axis=-1) - 1.0
        lower = np.where(residual > 0.0, log_den, lower)
        upper = np.where(residual <= 0.0, log_den, upper)
        slope = -1.0 / den - np.sum(occupancy * (den[..., None] / den_i), axis=-1)
        tolerance = 2e-13 * np.maximum(1.0, log_den)
        # A small occupancy residual is sufficient only when its slope also
        # bounds the error in log(D). Near complete saturation the slope can
        # vanish, so keep shrinking the bracket instead of exiting early.
        done = ((upper - lower <= tolerance) |
                (np.abs(residual) <= tolerance * np.abs(slope)))
        if np.all(done):
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            trial = log_den - residual / slope
        valid = np.isfinite(trial) & (trial > lower) & (trial < upper)
        log_den = np.where(done, log_den, np.where(valid, trial, 0.5 * (lower + upper)))
    den = np.exp(log_den)[..., None]
    c = total * den / (den + phase_capacity)
    q = H_eff * c / den
    return c, q


def _build_cpa_components(config: dict[str, Any], *, need_tdm_kinetics: bool) -> list[CPAComponent]:
    rows = active_components(config)
    # pH is an operating condition in the classic Process Setup tab, not a
    # resin/template parameter.  The 2021 CPA formulation supports pH-controlled
    # elution, so both EDM+CPA and TDM+CPA consume a programmed pH profile when
    # one is selected.
    _lp = config["process"]
    if not load_material_chemistry_is_explicit(_lp):
        _load_mode = str(_lp.get("load_mode") or "LINEAR").upper()
        _load_side = "end" if _load_mode == "STEP" else "start"
        _load_env = buffer_blend_chemistry(config, _lp.get(f"load_{_load_side}_percent_B"))
    else:
        _load_env = resolve_process_chemistry(
            config, source=_lp.get("load_source"), pH=_lp.get("load_pH"),
            salt_M=_lp.get("load_salt_concentration_M"),
            conductivity_mS_cm=_lp.get("load_conductivity_mS_cm"),
            salt_input_mode=_lp.get("load_salt_input_mode"),
        )
    pHA = float(_load_env["pH"])
    span = process_pH_span(config) if uses_cpa(config["model"]) else 0.0
    pH_varies = uses_cpa(config["model"]) and not is_hic(config) and span > 1e-9
    comps: list[CPAComponent] = []
    for row in rows:
        radius_nm = float(row["diameter_nm"]) / 2.0
        comps.append(
            CPAComponent(
                name=str(row["name"]),
                mass_fraction=float(row["mass_percent"]) / 100.0,
                radius_m=radius_nm * 1e-9,
                mw_kDa=_mw_from_radius_nm(radius_nm),
                As_m_inv=float(row["As_m_inv"]),
                Z_ref=0.0 if is_hic(config) else float(row["Z_ref"]),
                pH_ref=float(row["pH_ref"]) if pH_varies else pHA,
                Z_poly=(
                    float(row["Z1_per_pH"]) if pH_varies else 0.0,
                    float(row["Z2_per_pH2"]) if pH_varies else 0.0,
                    float(row["Z3_per_pH3"]) if pH_varies and span > 1.0 else 0.0,
                ),
                delta_ref=float(row["delta_ref"]),
                delta_pH_slope_m2_C=float(row["delta_pH_slope_m2_C"]) if pH_varies else 0.0,
                salt_sensitivity_per_M=float(row.get("salt_sensitivity_per_M", 1.0)),
                Z_lat=0.0,
                k_eff_m_s=float(row["k_eff_um_s"]) * 1e-6 if need_tdm_kinetics else 0.0,
                kkin_star_s=float(row["kkin_star_s"]) if need_tdm_kinetics else 0.0,
            )
        )
    return comps


def _cpa_model(config: dict[str, Any], comps: list[CPAComponent]) -> CPAModel:
    if is_hic(config):
        return CPAModel(comps, 0.0, 0.0, hic=True)
    ligand = float(config["cpa"]["ligand_surface_density_umol_m2"]) * 1e-6
    return CPAModel(comps, ligand, float(config["cpa"]["system_specific_adsorption_parameter"]))


def _column_geometry(config: dict[str, Any]) -> tuple[float, float, float]:
    col = config["column"]
    L = float(col["length_mm"]) * 1e-3
    V = float(col["volume_mL"]) * 1e-6
    A = V / L
    return L, V, A


def _numerical_resolution(config: dict[str, Any]) -> tuple[int, int]:
    """Return run-specific temporal and axial numerical resolution.

    time_steps is the requested number of intervals on the reported time grid and
    also caps the largest adaptive integration step at total_time/time_steps.
    Stiff/adaptive solvers may take additional internal steps when required.
    axial_positions is the number of finite-volume cells along the column.
    """
    n = config.get("numerics", {})
    time_steps = int(float(n.get("time_steps", 700)))
    axial_positions = int(float(n.get("axial_positions", 20)))
    return time_steps, axial_positions


def _simulate_tdm_langmuir(config: dict[str, Any]) -> dict[str, Any]:
    rows, H, b = _langmuir_arrays(config)
    names = [str(r["name"]) for r in rows]
    mass_frac = np.array([float(r["mass_percent"]) / 100.0 for r in rows])
    Dax = np.array([float(r["D_ax_mm2_s"]) * 1e-6 for r in rows], dtype=float)
    k_eff = np.array([float(r["k_eff_um_s"]) * 1e-6 for r in rows], dtype=float)
    epspi = np.array([float(r["accessible_particle_porosity"]) for r in rows], dtype=float)
    feed_vector = float(config["feed"]["total_concentration_mg_mL"]) * mass_frac
    program = _build_program(config, feed_vector)
    mobile_phase = MobilePhaseTransport(config, program, _scalar_transport)

    L, _V, A = _column_geometry(config)
    tdm = config["tdm"]
    epsv = float(tdm["void_fraction"])
    epsp = float(tdm["particle_porosity"])
    rp = float(tdm["bead_radius_um"]) * 1e-6
    Facc = epspi / epsp
    time_steps, nx = _numerical_resolution(config)
    dx = L / nx
    x_centres = (np.arange(nx) + 0.5) * dx
    salt_sensitivity = np.array([float(r["salt_sensitivity_per_M"]) for r in rows], dtype=float)
    n = len(rows)
    size = nx * n
    y0 = np.zeros(2 * size)

    def unpack(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return bulk concentration and conserved particle total concentration."""
        return y[:size].reshape(nx, n), y[size:].reshape(nx, n)

    beta = (1.0 - epsp) / np.maximum(Facc * epsp, SMALL)
    forcing_scale = 3.0 / rp * k_eff / np.maximum(Facc * epsp, SMALL)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        cb, particle_total = unpack(y)
        inlet, _, _, _ = program.at_time(t)
        current_cv = program.time_to_cv(t)
        Q = program.flow_at_time(t) * 1e-6 / 60.0
        u = Q / max(A * epsv, SMALL)
        dcb = np.zeros_like(cb)
        dparticle_total = np.zeros_like(particle_total)
        local_salts = mobile_phase.local(t, pore=True)[:, 1]
        cp, _q = _langmuir_c_q_from_total(
            particle_total, H, b, salt_M=local_salts,
            salt_sensitivity_per_M=salt_sensitivity, phase_ratio=beta,
        )
        for i in range(n):
            transfer = 3.0 / rp * k_eff[i] * (cb[:, i] - cp[:, i])
            dcb[:, i] = _scalar_transport(cb[:, i], inlet[i], u, Dax[i], dx) - (1.0 - epsv) / epsv * transfer
        for cell in range(nx):
            # The particle state is cp_i + beta_i*q_i. Integrating this
            # conserved quantity accounts for salt-driven equilibrium shifts.
            dparticle_total[cell] = forcing_scale * (cb[cell] - cp[cell])
        return np.concatenate([dcb.ravel(), dparticle_total.ravel()])

    solution = solve_ivp(
        rhs, (0.0, program.total_time_s), y0, method="BDF", rtol=1e-6, atol=1e-8,
        max_step=max(program.total_time_s / max(time_steps, 1), 1e-6), dense_output=True,
    )
    if not solution.success:
        raise RuntimeError("TDM+Competitive Langmuir integration failed: " + solution.message)

    sample_t = np.linspace(0.0, program.total_time_s, time_steps + 1)
    Y = solution.sol(sample_t)
    c_out = np.empty((len(sample_t), n))
    for k in range(len(sample_t)):
        cb, _cp = unpack(Y[:, k])
        c_out[k] = np.maximum(cb[-1], 0.0)
    final_cb, final_particle_total = unpack(solution.y[:, -1])
    final_cv = program.total_CV
    final_salts = mobile_phase.local(program.total_time_s, pore=True)[:, 1]
    final_cp, q_final = _langmuir_c_q_from_total(
        final_particle_total, H, b, salt_M=final_salts,
        salt_sensitivity_per_M=salt_sensitivity, phase_ratio=beta,
    )
    bed_cell_V_L = A * dx * 1000.0
    inventory_g = bed_cell_V_L * np.sum(
        epsv * final_cb + (1.0 - epsv) * epspi * final_particle_total,
        axis=0,
    )
    return {
        "program": program,
        "mobile_phase": mobile_phase,
        "component_names": names,
        "component_mw_kDa": None,
        "column_time_s": sample_t,
        "column_component_native": c_out,
        "column_pH": np.full(len(sample_t), np.nan),
        "column_ionic_strength_M": np.full(len(sample_t), np.nan),
        "final_inventory_native": inventory_g,
        "native_basis": "g/L",
        "diagnostics": {
            "transport_model": "TDM",
            "axial_transport_discretization": "conservative finite volume with monotonized-central slope-limited advection",
            "isotherm": "Competitive Langmuir",
            "axial_cells": nx,
            "requested_time_steps": time_steps,
            "internal_time_steps": max(len(solution.t) - 1, 0),
            "literature_TDM_symbols": "r_p, epsilon_v, epsilon_p, epsilon_p,i, F_acc,i, D_ax,i, k_eff,i, u_int",
            "competitive_langmuir_equation": "q_i*=qmax_i*b_i*exp(k_s_i*m_s)*c_i/(1+sum_j b_j*exp(k_s_j*m_s)*c_j)",
            "competitive_langmuir_H_relation": "H_i(0 M)=qmax_i*b_i; H_eff_i(salt)=qmax_i*b_eff_i(salt)",
            "hic_salt_model": "exponentially modified competitive Langmuir; salt concentration follows the shared buffer mixer and dispersive, pore-accessible column transport",
            "hic_salt_concentration_basis": "entered/resolved molarity used as molality surrogate",
            "hic_salt_sensitivity_per_M": salt_sensitivity.tolist(),
            "qmax_g_L_stationary_phase": [float(r["qmax_g_L"]) for r in rows],
            "number_of_competing_species": n,
        },
    }


def _wang_kinetic_rate(
    cp: np.ndarray, q: np.ndarray, salt_M: np.ndarray, pH: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Modified Wang Eq. (4), in g/L stationary phase per second.

    The paper's n_i is a species-specific stoichiometric exponent. q0 and eta
    have no species index and are shared. Salt is local pore salt, not the
    downstream conductivity detector's delayed reading.
    """
    rows = active_components(config)
    cp = np.maximum(np.asarray(cp, dtype=float), 0.0)
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    salt = np.maximum(np.asarray(salt_M, dtype=float), 0.0)[..., None]
    local_pH = np.asarray(pH, dtype=float)[..., None]
    values = lambda key: np.array([float(row[key]) for row in rows])
    kkin, keq, qmax, n = (values(key) for key in
                           ("wang_K_kin_s", "wang_k_eq", "wang_qmax_g_L", "wang_n"))
    beta0, beta1, beta2, beta3 = (values(key) for key in
                                    ("wang_beta0", "wang_beta1_per_M", "wang_beta2_L_g", "wang_beta3_per_pH"))
    q0 = float(config["wang"]["q0_g_L"])
    eta = float(config["wang"]["eta"])
    free = np.maximum(1.0 - np.sum(q / qmax, axis=-1, keepdims=True), 0.0)
    adsorption = keq * np.power(free, n) * np.power(cp, eta)
    environment = beta1 * salt + beta2 * cp + beta3 * local_pH
    exponent = 1.0 + n * beta0 * np.exp(np.clip(environment, -40.0, 40.0))
    with np.errstate(divide="ignore"):
        log_ratio = np.log(q / q0)
    log_desorption = (1.0 + n * beta0) * math.log(q0) + exponent * log_ratio
    desorption = np.where(q > 0.0, np.exp(np.clip(log_desorption, -80.0, 80.0)), 0.0)
    return (adsorption - desorption) / kkin


def _simulate_tdm_wang(config: dict[str, Any]) -> dict[str, Any]:
    rows = active_components(config)
    n = len(rows)
    names = [str(row["name"]) for row in rows]
    mass_frac = np.array([float(row["mass_percent"]) / 100.0 for row in rows])
    feed_vector = float(config["feed"]["total_concentration_mg_mL"]) * mass_frac
    program = _build_program(config, feed_vector)
    mobile_phase = MobilePhaseTransport(config, program, _scalar_transport)
    L, _V, A = _column_geometry(config)
    tdm = config["tdm"]
    epsv = float(tdm["void_fraction"])
    epsp = float(tdm["particle_porosity"])
    rp = float(tdm["bead_radius_um"]) * 1e-6
    Dax = np.array([float(row["D_ax_mm2_s"]) * 1e-6 for row in rows])
    k_eff = np.array([float(row["k_eff_um_s"]) * 1e-6 for row in rows])
    epspi = np.array([float(row["accessible_particle_porosity"]) for row in rows])
    time_steps, nx = _numerical_resolution(config)
    dx = L / nx
    size = nx * n
    y0 = np.zeros(3 * size)
    # BDF must differentiate the coupled protein states repeatedly during
    # refinement. Its Jacobian is local: axial transport spans nearby bulk
    # cells, while pore/adsorbed competition couples species in one cell.
    sparsity = lil_matrix((3 * size, 3 * size), dtype=int)
    for cell in range(nx):
        for species in range(n):
            index = cell * n + species
            for neighbor in range(max(0, cell - 2), min(nx, cell + 3)):
                sparsity[index, neighbor * n + species] = 1
            sparsity[index, size + index] = 1
            sparsity[size + index, index] = 1
            sparsity[size + index, size + index] = 1
            sparsity[2 * size + index, size + index] = 1
            for competing in range(n):
                q_index = 2 * size + cell * n + competing
                sparsity[size + index, q_index] = 1
                sparsity[2 * size + index, q_index] = 1

    def unpack(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (y[:size].reshape(nx, n), y[size:2*size].reshape(nx, n),
                y[2*size:].reshape(nx, n))

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        cb, cp, q = unpack(y)
        chemistry = mobile_phase.local(t, pore=True)
        inlet, _, _, _ = program.at_time(t)
        flow_m3_s = program.flow_at_time(t) * 1e-6 / 60.0
        u = flow_m3_s / max(A * epsv, SMALL)
        dq = _wang_kinetic_rate(cp, q, chemistry[:, 1], chemistry[:, 0], config)
        dcb = np.empty_like(cb)
        dcp = np.empty_like(cp)
        for i in range(n):
            transfer = 3.0 / rp * k_eff[i] * (cb[:, i] - cp[:, i])
            dcb[:, i] = (_scalar_transport(cb[:, i], inlet[i], u, Dax[i], dx)
                         - (1.0 - epsv) / epsv * transfer)
            dcp[:, i] = (transfer - (1.0 - epsp) * dq[:, i]) / epspi[i]
        return np.concatenate((dcb.ravel(), dcp.ravel(), dq.ravel()))

    solution = solve_ivp(
        rhs, (0.0, program.total_time_s), y0, method="BDF", rtol=2e-6, atol=1e-9,
        max_step=max(program.total_time_s / max(time_steps, 1), 1e-6), dense_output=True,
        jac_sparsity=sparsity.tocsr(),
    )
    if not solution.success:
        raise RuntimeError("TDM–Modified Wang integration failed: " + solution.message)
    sample_t = np.linspace(0.0, program.total_time_s, time_steps + 1)
    sampled = solution.sol(sample_t)
    c_out = np.empty((len(sample_t), n))
    for k in range(len(sample_t)):
        cb, _, _ = unpack(sampled[:, k])
        c_out[k] = np.maximum(cb[-1], 0.0)
    final_cb, final_cp, final_q = unpack(solution.y[:, -1])
    bed_cell_volume_L = A * dx * 1000.0
    inventory_g = bed_cell_volume_L * np.sum(
        epsv * final_cb + (1.0 - epsv) * epspi * final_cp
        + (1.0 - epsv) * (1.0 - epsp) * final_q, axis=0)
    return {
        "program": program, "mobile_phase": mobile_phase,
        "component_names": names, "component_mw_kDa": None,
        "column_time_s": sample_t, "column_component_native": c_out,
        "final_inventory_native": inventory_g, "native_basis": "g/L",
        "diagnostics": {
            "transport_model": "TDM", "isotherm": "Modified Wang (kinetic, Eq. 4)",
            "paper": "Beryamysoltan et al., J. Chromatogr. A 1783 (2026) 467108",
            "axial_cells": nx, "requested_time_steps": time_steps,
            "internal_time_steps": max(len(solution.t) - 1, 0),
            "wang_q0_g_L": config["wang"]["q0_g_L"],
            "wang_eta": config["wang"]["eta"],
            "wang_species_parameters": [
                {key: row[key] for key in (
                    "wang_K_kin_s", "wang_k_eq", "wang_qmax_g_L", "wang_n",
                    "wang_beta0", "wang_beta1_per_M", "wang_beta2_L_g", "wang_beta3_per_pH")}
                for row in rows],
        },
    }


def _simulate_tdm_cpa(config: dict[str, Any]) -> dict[str, Any]:
    comps = _build_cpa_components(config, need_tdm_kinetics=True)
    cpa_model = _cpa_model(config, comps)
    rows = active_components(config)
    feed_total_g_L = float(config["feed"]["total_concentration_mg_mL"])
    feed_vector = np.array([feed_total_g_L * c.mass_fraction / c.mw_kDa for c in comps], dtype=float)
    program = _build_program(config, feed_vector)
    mobile_phase = MobilePhaseTransport(config, program, _scalar_transport)

    L, _V, A = _column_geometry(config)
    tdm = config["tdm"]
    epsv = float(tdm["void_fraction"])
    epsp = float(tdm["particle_porosity"])
    rp = float(tdm["bead_radius_um"]) * 1e-6
    Dax = np.array([float(r["D_ax_mm2_s"]) * 1e-6 for r in rows], dtype=float)
    epspi = np.array([float(r["accessible_particle_porosity"]) for r in rows], dtype=float)
    Facc = epspi / epsp
    Dax_salt = float(config["system"]["salt_dispersion_mm2_s"]) * 1e-6
    # The attached study fixes k_eff to r_p/3 when film/pore resistance is negligible.
    k_eff_salt = rp / 3.0
    n = len(comps)
    time_steps, nx = _numerical_resolution(config)
    dx = L / nx
    x_centres = (np.arange(nx) + 0.5) * dx
    block = nx * n
    sl_cb = slice(0, block)
    sl_cp = slice(block, 2*block)
    sl_qv = slice(2*block, 3*block)
    y0 = np.zeros(3*block)

    def unpack(y: np.ndarray):
        return (
            y[sl_cb].reshape(nx, n),
            y[sl_cp].reshape(nx, n),
            y[sl_qv].reshape(nx, n),
        )

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        cb, cp, qv = unpack(y)
        local_chemistry = mobile_phase.local(t, pore=True)
        inlet_protein, _inlet_pH, inlet_I, _ = program.at_time(t)
        Q = program.flow_at_time(t) * 1e-6 / 60.0
        u = Q / max(A * epsv, SMALL)
        dqv = np.zeros_like(qv)
        current_cv = program.time_to_cv(t)
        for cell in range(nx):
            # pH is supplied as an operating profile.  In the absence of a
            # separate buffer-equilibrium model or proton-dispersion parameter,
            # propagate it as an unretained mobile-phase condition with the
            # interstitial convective delay to this axial cell.
            local_pH = float(local_chemistry[cell, 0])
            binding_salt = float(local_chemistry[cell, 1 if is_hic(config) else 2])
            dqv[cell] = cpa_model.kinetic_rate(cp[cell], qv[cell], local_pH, binding_salt)
        dcb = np.zeros_like(cb)
        dcp = np.zeros_like(cp)
        for i, comp in enumerate(comps):
            transfer = 3.0 / rp * comp.k_eff_m_s * (cb[:, i] - cp[:, i])
            dcb[:, i] = _scalar_transport(cb[:, i], inlet_protein[i], u, Dax[i], dx) - (
                (1.0 - epsv) / epsv * transfer
            )
            denom = max(Facc[i] * epsp, SMALL)
            dcp[:, i] = (
                3.0 / rp * comp.k_eff_m_s / denom * (cb[:, i] - cp[:, i])
                - (1.0 - epsp) / denom * dqv[:, i]
            )

        return np.concatenate([dcb.ravel(), dcp.ravel(), dqv.ravel()])

    solution = solve_ivp(
        rhs, (0.0, program.total_time_s), y0, method="BDF", rtol=2e-6, atol=1e-9,
        max_step=max(program.total_time_s / max(time_steps, 1), 1e-6), dense_output=True,
    )
    if not solution.success:
        raise RuntimeError("TDM+CPA integration failed: " + solution.message)

    sample_t = np.linspace(0.0, program.total_time_s, time_steps + 1)
    Y = solution.sol(sample_t)
    cb_out = np.empty((len(sample_t), n))
    ionic_out = np.empty(len(sample_t))
    pH_out = np.empty(len(sample_t))
    for k in range(len(sample_t)):
        cb, _, _ = unpack(Y[:, k])
        cb_out[k] = np.maximum(cb[-1], 0.0)
        chemistry = mobile_phase.local(float(sample_t[k]))[-1]
        ionic_out[k] = chemistry[2]
        pH_out[k] = chemistry[0]
    final_cb, final_cp, final_qv = unpack(solution.y[:, -1])
    bed_cell_V = A * dx
    inventory_mol = bed_cell_V * np.sum(
        epsv * final_cb
        + (1.0 - epsv) * epspi * final_cp
        + (1.0 - epsv) * (1.0 - epsp) * final_qv,
        axis=0,
    )
    return {
        "program": program,
        "mobile_phase": mobile_phase,
        "component_names": [c.name for c in comps],
        "component_mw_kDa": np.array([c.mw_kDa for c in comps]),
        "column_time_s": sample_t,
        "column_component_native": cb_out,
        "column_pH": pH_out,
        "column_ionic_strength_M": ionic_out,
        "final_inventory_native": inventory_mol,
        "native_basis": "mol/m3",
        "diagnostics": {
            "transport_model": "TDM",
            "axial_transport_discretization": "conservative finite volume with monotonized-central slope-limited advection",
            "isotherm": "CPA",
            "axial_cells": nx,
            "requested_time_steps": time_steps,
            "internal_time_steps": max(len(solution.t) - 1, 0),
            "literature_TDM_symbols": "r_p, epsilon_v, epsilon_p, epsilon_p,i, F_acc,i, D_ax,i, D_ax,salt, k_eff,i, u_int",
            "salt_k_eff_rule": "k_eff,salt = r_p/3 (study assumption)",
            "pH_rule": "user-programmed pH profile with interstitial convective delay; pH-dependent CPA parameters follow Briskot et al. (2021)",
            "radius_rule": "diameter_nm / 2",
            "fixed_temperature_K": CPA_TEMPERATURE_K,
            "fixed_relative_permittivity": CPA_RELATIVE_PERMITTIVITY,
            "fixed_CEX_ligand_pK": CPA_CEX_LIGAND_PK,
        },
    }


def _simulate_edm_langmuir(config: dict[str, Any]) -> dict[str, Any]:
    rows, H, b = _langmuir_arrays(config)
    names = [str(r["name"]) for r in rows]
    mass_frac = np.array([float(r["mass_percent"]) / 100.0 for r in rows])
    D = np.array([float(r["D_app_mm2_s"]) * 1e-6 for r in rows])
    feed_vector = float(config["feed"]["total_concentration_mg_mL"]) * mass_frac
    program = _build_program(config, feed_vector)
    mobile_phase = MobilePhaseTransport(config, program, _scalar_transport)

    L, _V, A = _column_geometry(config)
    eps = float(config["edm"]["total_bed_porosity"])
    salt_sensitivity = np.array([float(r["salt_sensitivity_per_M"]) for r in rows], dtype=float)
    F = (1.0 - eps) / eps
    time_steps, nx = _numerical_resolution(config)
    dx = L / nx
    x_centres = (np.arange(nx) + 0.5) * dx
    n = len(rows)
    y0 = np.zeros(nx*n)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        total = np.maximum(y.reshape(nx, n), 0.0)
        inlet, _, _, _ = program.at_time(t)
        Q = program.flow_at_time(t) * 1e-6 / 60.0
        u = Q / max(A * eps, SMALL)
        current_cv = program.time_to_cv(t)
        local_salts = mobile_phase.local(t)[:, 1]
        c, _q = _langmuir_c_q_from_total(
            total, H, b, salt_M=local_salts,
            salt_sensitivity_per_M=salt_sensitivity, phase_ratio=F,
        )
        forcing = np.zeros_like(c)
        for i in range(n):
            forcing[:, i] = _scalar_transport(c[:, i], inlet[i], u, D[i], dx)
        # total = c + F*q remains the conserved state as salt changes.
        return forcing.ravel()

    solution = solve_ivp(
        rhs, (0.0, program.total_time_s), y0, method="BDF", rtol=1e-6, atol=1e-8,
        max_step=max(program.total_time_s / max(time_steps, 1), 1e-6), dense_output=True,
    )
    if not solution.success:
        raise RuntimeError("EDM+Competitive Langmuir integration failed: " + solution.message)
    sample_t = np.linspace(0.0, program.total_time_s, time_steps + 1)
    Y = solution.sol(sample_t)
    c_out = np.empty((len(sample_t), n))
    for k in range(len(sample_t)):
        total_out = np.maximum(Y[:, k].reshape(nx, n)[-1], 0.0)
        current_cv = program.time_to_cv(float(sample_t[k]))
        source_cv = max(current_cv - eps * x_centres[-1] / L, 0.0)
        local_salt_M = mobile_phase.local(float(sample_t[k]))[-1, 1]
        c_out[k], _q_out = _langmuir_c_q_from_total(
            total_out, H, b, salt_M=local_salt_M,
            salt_sensitivity_per_M=salt_sensitivity, phase_ratio=F,
        )
    final_total = np.maximum(solution.y[:, -1].reshape(nx, n), 0.0)
    final_cv = program.total_CV
    final_salts = mobile_phase.local(program.total_time_s)[:, 1]
    final_c, q_final = _langmuir_c_q_from_total(
        final_total, H, b, salt_M=final_salts,
        salt_sensitivity_per_M=salt_sensitivity, phase_ratio=F,
    )
    bed_cell_V_L = A * dx * 1000.0
    inventory_g = bed_cell_V_L * np.sum(eps * final_c + (1.0 - eps) * q_final, axis=0)
    return {
        "program": program,
        "mobile_phase": mobile_phase,
        "component_names": names,
        "component_mw_kDa": None,
        "column_time_s": sample_t,
        "column_component_native": c_out,
        "column_pH": np.full(len(sample_t), np.nan),
        "column_ionic_strength_M": np.full(len(sample_t), np.nan),
        "final_inventory_native": inventory_g,
        "native_basis": "g/L",
        "diagnostics": {
            "transport_model": "EDM",
            "axial_transport_discretization": "conservative finite volume with monotonized-central slope-limited advection",
            "isotherm": "Competitive Langmuir",
            "axial_cells": nx,
            "requested_time_steps": time_steps,
            "internal_time_steps": max(len(solution.t) - 1, 0),
            "competitive_langmuir_equation": "q_i*=qmax_i*b_i*exp(k_s_i*m_s)*c_i/(1+sum_j b_j*exp(k_s_j*m_s)*c_j)",
            "competitive_langmuir_H_relation": "H_i(0 M)=qmax_i*b_i; H_eff_i(salt)=qmax_i*b_eff_i(salt)",
            "hic_salt_model": "exponentially modified competitive Langmuir; salt concentration follows the shared buffer mixer and dispersive, pore-accessible column transport",
            "hic_salt_concentration_basis": "entered/resolved molarity used as molality surrogate",
            "hic_salt_sensitivity_per_M": salt_sensitivity.tolist(),
            "qmax_g_L_stationary_phase": [float(r["qmax_g_L"]) for r in rows],
            "number_of_competing_species": n,
        },
    }


def _simulate_edm_cpa(config: dict[str, Any]) -> dict[str, Any]:
    comps = _build_cpa_components(config, need_tdm_kinetics=False)
    cpa_model = _cpa_model(config, comps)
    feed_total_g_L = float(config["feed"]["total_concentration_mg_mL"])
    feed_vector = np.array([feed_total_g_L * c.mass_fraction / c.mw_kDa for c in comps], dtype=float)
    program = _build_program(config, feed_vector)
    mobile_phase = MobilePhaseTransport(config, program, _scalar_transport)

    rows = active_components(config)
    D = np.array([float(r["D_app_mm2_s"]) * 1e-6 for r in rows])
    L, _V, A = _column_geometry(config)
    eps = float(config["edm"]["total_bed_porosity"])
    Fphase = (1.0 - eps) / eps
    time_steps, nx = _numerical_resolution(config)
    dx = L / nx
    n = len(comps)
    x_centres = (np.arange(nx) + 0.5) * dx

    def env_at(t: float, cell: int) -> tuple[float, float]:
        chemistry = mobile_phase.local(t)[cell]
        return float(chemistry[0]), float(chemistry[1 if is_hic(config) else 2])

    total = np.zeros((nx, n))  # c + F q, conserved by the EDM transport balance
    c = np.zeros_like(total)
    q = np.zeros_like(total)

    maxD = max(float(np.max(D)), 0.0)
    max_flow = max(stage.flow_mL_min for stage in program.stages) * 1e-6 / 60.0
    max_u = max_flow / max(A * eps, SMALL)
    rate = abs(max_u) / dx + 2.0 * maxD / (dx*dx)
    base_dt = 0.35 / max(rate, 1e-12)
    base_dt = min(base_dt, program.cv_time_s / 60.0, program.total_time_s / max(time_steps, 1))

    times = [0.0]
    outs = [np.zeros(n)]
    outlet_amount_native = np.zeros(n)  # mol, accumulated with the exact finite-volume outlet flux
    pH0, I0, _B0, _a, _b = program.environment_with_derivative(0.0)
    pHs = [pH0]
    Is = [I0]
    t = 0.0
    stage_boundaries = program.stage_boundaries_s
    while t < program.total_time_s - 1e-12:
        dt = min(base_dt, program.total_time_s - t)
        for boundary in stage_boundaries:
            if t + 1e-12 < boundary < t + dt - 1e-12:
                dt = boundary - t
                break
        midpoint = t + 0.5*dt
        inlet, _, _, _ = program.at_time(midpoint)
        Q = program.flow_at_time(midpoint) * 1e-6 / 60.0
        u = Q / max(A * eps, SMALL)
        outlet_amount_native += Q * np.maximum(c[-1], 0.0) * dt
        forcing = np.zeros_like(c)
        for i in range(n):
            forcing[:, i] = _scalar_transport(c[:, i], inlet[i], u, D[i], dx)
        total = np.maximum(total + dt * forcing, 0.0)
        t += dt
        for cell in range(nx):
            pH, ionic = env_at(t, cell)
            c[cell], q[cell] = cpa_model.equilibrium_from_total(total[cell], Fphase, pH, ionic)
        times.append(t)
        outs.append(np.maximum(c[-1].copy(), 0.0))
        chemistry = mobile_phase.local(t)[-1]
        pHs.append(chemistry[0]); Is.append(chemistry[2])

    raw_t = np.asarray(times)
    raw_out = np.asarray(outs)
    sample_t = np.linspace(0.0, program.total_time_s, time_steps + 1)
    c_out = np.column_stack([np.interp(sample_t, raw_t, raw_out[:, i]) for i in range(n)])
    pH_out = np.interp(sample_t, raw_t, np.asarray(pHs))
    I_out = np.interp(sample_t, raw_t, np.asarray(Is))

    bed_cell_V = A * dx
    inventory_mol = bed_cell_V * np.sum(eps*c + (1.0-eps)*q, axis=0)
    return {
        "program": program,
        "mobile_phase": mobile_phase,
        "component_names": [comp.name for comp in comps],
        "component_mw_kDa": np.array([comp.mw_kDa for comp in comps]),
        "column_time_s": sample_t,
        "column_component_native": c_out,
        "mass_balance_time_s": raw_t,
        "mass_balance_component_native": raw_out,
        "outlet_amount_native": outlet_amount_native,
        "column_pH": pH_out,
        "column_ionic_strength_M": I_out,
        "final_inventory_native": inventory_mol,
        "native_basis": "mol/m3",
        "diagnostics": {
            "transport_model": "EDM",
            "axial_transport_discretization": "conservative finite volume with monotonized-central slope-limited advection",
            "isotherm": "CPA", "axial_cells": nx,
            "requested_time_steps": time_steps,
            "internal_time_steps": max(len(raw_t) - 1, 0),
            "EDM_time_integrator": "conservative explicit finite-volume update of c+Fq with algebraic CPA equilibrium inversion",
            "radius_rule": "diameter_nm / 2",
            "buffer_transport": "shared upstream buffer mixer and dispersive column salt transport",
            "fixed_temperature_K": CPA_TEMPERATURE_K,
            "fixed_relative_permittivity": CPA_RELATIVE_PERMITTIVITY,
            "fixed_CEX_ligand_pK": CPA_CEX_LIGAND_PK,
        },
    }


def _output_trace(config: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    program: StageProgram = sim["program"]
    t = np.asarray(sim["column_time_s"])
    native = np.maximum(np.asarray(sim["column_component_native"]), 0.0)
    if sim["native_basis"] == "mol/m3":
        mw = np.asarray(sim["component_mw_kDa"])
        component_g_L = native * mw[None, :]
    else:
        component_g_L = native
    column_component_g_L = component_g_L.copy()
    cv = np.asarray([program.time_to_cv(float(x)) for x in t], dtype=float)
    system = config["system"]
    component_g_L = delay_by_volume(cv, component_g_L, float(system["uv_dead_volume_mL"]) / float(config["column"]["volume_mL"]))
    volume_mL = cv * float(config["column"]["volume_mL"])
    out = {
        "t_s": t,
        "time_min": t / 60.0,
        "CV": cv,
        "volume_mL": volume_mL,
        "column_component_g_L": column_component_g_L,
        "column_total_g_L": np.sum(column_component_g_L, axis=1),
        "component_g_L": component_g_L,
        "total_g_L": np.sum(component_g_L, axis=1),
    }
    uv_factors = np.asarray([
        float(config["conversion"]["uv_to_protein_mAU_L_g"]) if i == 0 else
        float(row.get("uv_response_factor_mAU_L_g") or config["conversion"]["uv_to_protein_mAU_L_g"])
        for i, row in enumerate(active_components(config))
    ], dtype=float)
    out["uv_response_factors_mAU_L_g"] = uv_factors
    out["component_uv_mAU"] = component_g_L * uv_factors[None, :]
    out["uv_mAU"] = np.sum(out["component_uv_mAU"], axis=1)
    chemistry = sim["mobile_phase"].outlet(t)
    # Programmed %B is the pump command at this point in the recipe. Column %B
    # and conductivity follow the physical mixer/column response.
    out["programmed_percent_B"] = np.asarray([program.percent_B_at_cv(float(v)) for v in cv])
    out["percent_B"] = out["programmed_percent_B"].copy()
    out["column_percent_B"] = chemistry[:, 4]
    out["pH"] = chemistry[:, 0]
    out["salt_concentration_M"] = chemistry[:, 1]
    out["ionic_strength_M"] = chemistry[:, 2]
    binding = np.asarray([sim["mobile_phase"].local(float(time), pore=uses_tdm(config["model"]))[-1] for time in t])
    out["binding_salt_concentration_M"] = binding[:, 1]
    out["binding_ionic_strength_M"] = binding[:, 2]
    if uses_langmuir(config["model"]):
        # The solver uses the same local binding-salt profile in its column
        # equations. Export affinity at the last cell so b and k_s can be
        # audited independently of detector delay and UV peak position.
        components = active_components(config)
        b = np.asarray([float(row["b_L_g"]) for row in components])
        k_s = np.asarray([float(row["salt_sensitivity_per_M"]) for row in components])
        out["component_effective_b_L_g"] = b[None, :] * np.exp(np.clip(
            binding[:, 1, None] * k_s[None, :], -MAX_HIC_LOG_AFFINITY, MAX_HIC_LOG_AFFINITY,
        ))
    out["programmed_salt_concentration_M"] = np.asarray([program.chemistry_at_cv(float(v))[1] for v in cv])
    out["column_conductivity_mS_cm"] = chemistry[:, 3]
    out["programmed_conductivity_mS_cm"] = np.asarray([program.chemistry_at_cv(float(v))[3] for v in cv])
    out["conductivity_mS_cm"] = delay_by_volume(
        cv, chemistry[:, 3], float(system["conductivity_dead_volume_mL"]) / float(config["column"]["volume_mL"]),
        initial=float(chemistry[0, 3]),
    )
    # The flow trace is the actual stage command (load, PLW and elution),
    # expressed at every reported time point so output plots and CSV rows use
    # the same value that drove the transport solver.
    out["flow_mL_min"] = np.asarray([program.flow_at_time(float(time)) for time in t], dtype=float)
    out["stage_index"] = np.asarray([program._stage_index_cv(float(v)) + 1 for v in cv], dtype=float)

    return out


def _mass_balance(config: dict[str, Any], sim: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    program: StageProgram = sim["program"]
    names = sim["component_names"]
    rows = active_components(config)
    Vcol_mL = float(config["column"]["volume_mL"])
    load_density = float(config["feed"]["load_density_mg_mL_resin"])
    input_g = np.array([load_density * Vcol_mL / 1000.0 * float(r["mass_percent"]) / 100.0 for r in rows])
    flow_L_s = program.flow_array_L_s(np.asarray(trace["t_s"], dtype=float))
    if "outlet_amount_native" in sim:
        if sim["native_basis"] == "mol/m3":
            out_g = np.asarray(sim["outlet_amount_native"]) * np.asarray(sim["component_mw_kDa"]) * 1000.0
        else:
            out_g = np.asarray(sim["outlet_amount_native"])
    elif "mass_balance_time_s" in sim:
        mb_t = np.asarray(sim["mass_balance_time_s"])
        mb_native = np.asarray(sim["mass_balance_component_native"])
        if sim["native_basis"] == "mol/m3":
            mb_g_L = mb_native * np.asarray(sim["component_mw_kDa"])[None, :]
        else:
            mb_g_L = mb_native
        out_g = np.array([_trapezoidal_integral(mb_g_L[:, i] * program.flow_array_L_s(np.asarray(mb_t, dtype=float)), mb_t) for i in range(len(names))])
    else:
        out_g = np.array([_trapezoidal_integral(trace["column_component_g_L"][:, i] * flow_L_s, trace["t_s"]) for i in range(len(names))])
    if sim["native_basis"] == "mol/m3":
        inv_g = np.asarray(sim["final_inventory_native"]) * np.asarray(sim["component_mw_kDa"]) * 1000.0
    else:
        inv_g = np.asarray(sim["final_inventory_native"])
    error_g = input_g - out_g - inv_g
    relative = np.divide(error_g, input_g, out=np.zeros_like(error_g), where=input_g > 0)
    return {
        name: {
            "input_g": float(input_g[i]),
            "outlet_g": float(out_g[i]),
            "column_inventory_g": float(inv_g[i]),
            "closure_error_g": float(error_g[i]),
            "closure_error_percent_of_input": float(relative[i] * 100.0),
        }
        for i, name in enumerate(names)
    }


def _trapezoidal_integral(values: np.ndarray, x: np.ndarray) -> float:
    """Integrate arrays across NumPy versions where trapz may be removed."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz", None)
    if integrate is None:
        raise RuntimeError("Mass balance needs numpy.trapezoid or numpy.trapz for integration.")
    return float(integrate(values, x))


def _nominal_langmuir_saturation(config: dict[str, Any]) -> dict[str, float] | None:
    """Compare target feed dose with the stationary-phase saturation ceiling.

    This is a capacity check, not a dynamic binding-capacity prediction. In a
    competitive run another protein may occupy part of the target's capacity.
    """
    if not uses_langmuir(config["model"]):
        return None
    if config["model"] == MODEL_EDM_LANGMUIR:
        stationary_fraction = 1.0 - float(config["edm"]["total_bed_porosity"])
    else:
        tdm = config["tdm"]
        stationary_fraction = (1.0 - float(tdm["void_fraction"])) * (1.0 - float(tdm["particle_porosity"]))
    target = active_components(config)[0]
    capacity = stationary_fraction * float(target["qmax_g_L"])
    dose = float(config["feed"]["load_density_mg_mL_resin"]) * float(target["mass_percent"]) / 100.0
    return {
        "stationary_fraction_of_bed": stationary_fraction,
        "target_qmax_g_L_stationary_phase": float(target["qmax_g_L"]),
        "target_nominal_saturation_g_L_bed": capacity,
        "target_feed_load_g_L_bed": dose,
        "target_load_to_saturation_ratio": dose / capacity,
    }


def _langmuir_affinity_receipt(config: dict[str, Any], stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Show the salt-dependent affinity consumed at programmed stage endpoints."""
    if not uses_langmuir(config["model"]):
        return []
    result = []
    for component in active_components(config):
        b = float(component["b_L_g"])
        k_s = float(component["salt_sensitivity_per_M"])
        result.append({
            "component": str(component["name"]),
            "b_L_g": b,
            "salt_sensitivity_per_M": k_s,
            "qmax_g_L_stationary_phase": float(component["qmax_g_L"]),
            "stage_endpoints": [
                {
                    "stage": stage["name"],
                    "start_salt_M": stage["start_salt_M"],
                    "end_salt_M": stage["end_salt_M"],
                    "start_b_eff_L_g": b * math.exp(float(np.clip(k_s * stage["start_salt_M"], -MAX_HIC_LOG_AFFINITY, MAX_HIC_LOG_AFFINITY))),
                    "end_b_eff_L_g": b * math.exp(float(np.clip(k_s * stage["end_salt_M"], -MAX_HIC_LOG_AFFINITY, MAX_HIC_LOG_AFFINITY))),
                }
                for stage in stages
            ],
        })
    return result


def simulate(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_config(config)
    errors = validate_config(config, strict=True)
    if errors:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(errors))
    model = config["model"]
    if model == MODEL_TDM_WANG:
        sim = _simulate_tdm_wang(config)
    elif model == MODEL_TDM_CPA:
        sim = _simulate_tdm_cpa(config)
    elif model == MODEL_TDM_LANGMUIR:
        sim = _simulate_tdm_langmuir(config)
    elif model == MODEL_EDM_CPA:
        sim = _simulate_edm_cpa(config)
    elif model == MODEL_EDM_LANGMUIR:
        sim = _simulate_edm_langmuir(config)
    else:
        raise ValueError(f"Unsupported model: {model}")
    trace = _output_trace(config, sim)
    program: StageProgram = sim["program"]
    first_elution_CV = next(
        (program.stage_start_CV[i] for i, stage in enumerate(program.stages) if stage.kind == "ELUTION"),
        program.total_CV,
    )
    target_detector = np.asarray(trace["component_g_L"])[:, 0]
    target_column = np.asarray(trace["column_component_g_L"])[:, 0]
    detector_index = int(np.argmax(target_detector))
    column_index = int(np.argmax(target_column))
    detector_peak_CV = float(trace["CV"][detector_index]) if target_detector[detector_index] > 1e-12 else None
    target_peak = {
        "detector_peak_CV": detector_peak_CV,
        "detector_peak_g_L": float(target_detector[detector_index]),
        "column_outlet_peak_CV": float(trace["CV"][column_index]) if target_column[column_index] > 1e-12 else None,
        "first_elution_start_CV": float(first_elution_CV),
        "CV_after_elution_start": (detector_peak_CV - first_elution_CV if detector_peak_CV is not None else None),
        "stage_at_detector_peak": (
            program.stages[program._stage_index_time(float(trace["t_s"][detector_index]))].name
            if detector_peak_CV is not None else None
        ),
        "binding_salt_M_at_detector_peak": (
            float(trace["binding_salt_concentration_M"][detector_index]) if detector_peak_CV is not None else None
        ),
    }
    load_capacity = float(config["feed"]["load_density_mg_mL_resin"])
    required_load_CV = float(program.stages[0].duration_CV)
    required_load_volume_mL = required_load_CV * float(config["column"]["volume_mL"])
    stage_chemistry_used = _stage_receipt(config, program)
    affinity_used = _langmuir_affinity_receipt(config, stage_chemistry_used)
    nominal_saturation = _nominal_langmuir_saturation(config)
    parameter_receipt = {
        "required_inputs_used": required_parameter_values(config),
        "derived_species_feed_concentration_mg_mL": {
            str(r["name"]): float(config["feed"]["total_concentration_mg_mL"]) * float(r["mass_percent"]) / 100.0
            for r in active_components(config)
        },
        "load_input_used": {
            "specified_basis": normalize_load_amount_basis(config["process"].get("load_amount_basis")),
            "capacity_g_L_resin": load_capacity,
            "feed_concentration_mg_mL": float(config["feed"]["total_concentration_mg_mL"]),
            "required_feed_volume_CV": required_load_CV,
            "required_feed_volume_mL": required_load_volume_mL,
            "total_protein_input_g": load_capacity * float(config["column"]["volume_mL"]) / 1000.0,
            "material_chemistry_mode": str(config["process"].get("load_material_chemistry_mode") or "BUFFER_RECIPE"),
            "material_chemistry_source": (
                str(config["process"].get("load_source") or "DIRECT")
                if load_material_chemistry_is_explicit(config["process"])
                else "BUFFER_BLEND"
            ),
            "material_salt_input_mode": stage_chemistry_used[0]["start_salt_input_mode"],
            "material_pH": float(program.stages[0].start_pH),
            "material_salt_M": float(program.stages[0].start_salt_M),
            "material_ionic_strength_M": float(program.stages[0].start_I_M),
            "material_conductivity_mS_cm": float(program.stages[0].start_conductivity_mS_cm),
            "nominal_langmuir_saturation": nominal_saturation,
        },
        "batch_shared_parameter_fingerprint": batch_parameter_fingerprint(config),
        "buffer_chemistry_used": {key: buffer_chemistry(config, config["process"].get(key, {}))
                                  for key in ("buffer_A", "buffer_B")},
        "stage_chemistry_used": stage_chemistry_used,
        "langmuir_affinity_used": affinity_used,
        "target_peak": target_peak,
    }
    # Keep the receipt next to solver diagnostics so a saved result records the
    # exact visible/required parameter values consumed by this run.
    sim["diagnostics"]["batch_shared_parameter_fingerprint"] = parameter_receipt["batch_shared_parameter_fingerprint"]
    sim["diagnostics"]["chromatography_mode"] = config["chromatography_mode"]
    sim["diagnostics"]["system_response"] = dict(config["system"])
    sim["diagnostics"]["column_salt_transport"] = "stirred upstream buffer volume; conservative axial transport; TDM salt fully pore-accessible; column initially equilibrated to effective load chemistry"
    sim["diagnostics"]["binding_chemistry"] = ("Analytical salt molarity [M]" if is_hic(config) or uses_langmuir(model)
                                               else "Ionic strength [M]")
    sim["diagnostics"]["target_peak"] = target_peak
    if uses_cpa(model) and is_hic(config):
        sim["diagnostics"]["CPA_HIC_extension"] = "K_i = Delta_i * available_surface_i(q) * exp(k_s_i * salt_M); empirical hydrophobic affinity with CPA excluded-area competition, not the paper's IEX electrostatic law"
    warnings = []
    if nominal_saturation is not None:
        sim["diagnostics"]["nominal_langmuir_saturation"] = nominal_saturation
        if nominal_saturation["target_load_to_saturation_ratio"] > 1.0 + 1e-9:
            warnings.append(
                "Target load exceeds nominal stationary-phase saturation: "
                f"{nominal_saturation['target_feed_load_g_L_bed']:g} g/L packed bed loaded versus "
                f"{nominal_saturation['target_nominal_saturation_g_L_bed']:g} g/L packed bed "
                "(qmax × stationary-phase bed fraction). Breakthrough can occur during loading "
                "or washing even when high salt strongly favors binding."
            )
    if is_hic(config):
        load_salt = program.stages[0].chemistry(0.0)[1]
        elution_salts = [stage.chemistry(f)[1] for stage in program.stages if stage.kind == "ELUTION" for f in (0.0, 1.0)]
        if elution_salts and min(elution_salts) >= load_salt:
            warnings.append("The elution program does not reduce salt below the loading value. Check the selected chemistry input and Buffer A/B direction for HIC release.")
        elution_start_s = next((program.stage_start_time_s[i] for i, stage in enumerate(program.stages) if stage.kind == "ELUTION"), program.total_time_s)
        mask = np.asarray(trace["t_s"]) <= elution_start_s
        flow = program.flow_array_L_s(np.asarray(trace["t_s"])[mask])
        escaped = _trapezoidal_integral(np.asarray(trace["column_total_g_L"])[mask] * flow, np.asarray(trace["t_s"])[mask])
        input_g = load_capacity * float(config["column"]["volume_mL"]) / 1000.0
        sim["diagnostics"]["protein_outlet_before_elution_percent_of_load"] = 100.0 * escaped / input_g
        if escaped > 0.01 * input_g:
            warnings.append(
                f"{100.0 * escaped / input_g:.3g}% of loaded protein exits before the elution stage "
                "(breakthrough/wash loss, not an elution-stage peak). Check loading versus "
                "stationary-phase capacity, salt/%B, affinity and TDM mass-transfer rates."
            )
    sim["diagnostics"]["warnings"] = warnings
    return {
        "simulation": sim,
        "trace": trace,
        "mass_balance": _mass_balance(config, sim, trace),
        "parameter_receipt": parameter_receipt,
    }


def _stage_receipt(config: dict[str, Any], program: StageProgram) -> list[dict[str, Any]]:
    """Describe the exact stage sequence used by the solver in cumulative CV."""
    process = config["process"]
    blend_modes = (
        f"Buffer A={salt_input_mode(process['buffer_A'])}; "
        f"Buffer B={salt_input_mode(process['buffer_B'])}"
    )

    def provenance(source: Any, direct_mode: Any) -> tuple[str, str]:
        src = str(source or "DIRECT").upper()
        if src in {"BUFFER_A", "BUFFER_B"}:
            buffer = process["buffer_A" if src == "BUFFER_A" else "buffer_B"]
            return src, salt_input_mode(buffer)
        return "DIRECT", salt_input_mode({"salt_input_mode": direct_mode})

    counters = {"PLW": 0, "ELUTION": 0}
    rows: list[dict[str, Any]] = []
    for index, stage in enumerate(program.stages):
        start_b: Any = stage.start_percent_B
        end_b: Any = stage.end_percent_B
        if stage.kind == "LOAD":
            control = "EXPLICIT_LOAD_MATERIAL" if load_material_chemistry_is_explicit(process) else "BUFFER_B_PERCENT"
            if control == "EXPLICIT_LOAD_MATERIAL":
                start_source, start_mode = provenance(process.get("load_source"), process.get("load_salt_input_mode"))
                end_source, end_mode = start_source, start_mode
            else:
                start_source = end_source = "BUFFER_BLEND"
                start_mode = end_mode = blend_modes
        else:
            control = "LOAD"
        if stage.kind in counters:
            counters[stage.kind] += 1
            key = "plw_steps" if stage.kind == "PLW" else "elution_steps"
            raw = process[key][counters[stage.kind] - 1]
            control = str(raw.get("chemistry_control") or "BUFFER_B_PERCENT").upper()
            if control == "BUFFER_B_PERCENT":
                start_source = end_source = "BUFFER_BLEND"
                start_mode = end_mode = blend_modes
            else:
                start_source, start_mode = provenance(raw.get("start_source"), raw.get("start_salt_input_mode"))
                end_source, end_mode = provenance(raw.get("end_source"), raw.get("end_salt_input_mode"))
        rows.append({
            "marker": index + 1,
            "name": stage.name,
            "kind": stage.kind,
            "start_CV": float(program.stage_start_CV[index]),
            "end_CV": float(program.stage_end_CV[index]),
            "duration_CV": float(stage.duration_CV),
            "flow_mL_min": float(stage.flow_mL_min),
            "mode": str(stage.mode).upper(),
            "chemistry_control": control,
            "start_salt_source": start_source,
            "end_salt_source": end_source,
            "start_salt_input_mode": start_mode,
            "end_salt_input_mode": end_mode,
            "start_percent_B": start_b,
            "end_percent_B": end_b,
            "start_pH": float(stage.start_pH),
            "end_pH": float(stage.end_pH),
            "start_salt_M": float(stage.start_salt_M),
            "end_salt_M": float(stage.end_salt_M),
            "start_ionic_strength_M": float(stage.start_I_M),
            "end_ionic_strength_M": float(stage.end_I_M),
            "start_conductivity_mS_cm": float(stage.start_conductivity_mS_cm),
            "end_conductivity_mS_cm": float(stage.end_conductivity_mS_cm),
        })
    return rows


def _svg_plot(
    x: np.ndarray, series: list[tuple[str, np.ndarray]], *, stages=None, series_x=None,
    percent_B=None, column_percent_B=None, conductivity=None, time_min=None,
    flow_mL_min=None, total_cv=None, show_percent_B=True, show_conductivity=True,
    show_flow_rate=True,
    width=1100, height=650, y_axis_label="UV absorbance (mAU)",
) -> str:
    """Plot a full program, with optional independent sample CVs for each signal."""
    if series_x is None:
        series_x = [x] * len(series)
    if len(series_x) != len(series) or any(
        len(sample_cv) != len(values)
        for sample_cv, (_, values) in zip(series_x, series)
    ):
        raise ValueError("Each plotted signal must have one CV coordinate per sample.")
    left, right, top, bottom = 86, 270, 170, 520
    pw, ph = width - left - right, bottom - top
    # Neither a caller's display limit nor a sparse signal grid may cut off
    # any enabled process step. All chart coordinates use cumulative run CV.
    xmax = max(float(total_cv or 0.0), float(np.max(x)),
               max((float(stage["end_CV"]) for stage in stages or []), default=0.0), 1e-12)
    command_cv, command_b, command_flow = [], [], []
    for stage in stages or []:
        command_cv.extend([stage["start_CV"], stage["end_CV"]])
        start_b, end_b = (float(stage[key]) if stage[key] is not None else math.nan
                          for key in ("start_percent_B", "end_percent_B"))
        if stage["mode"] == "STEP":
            start_b = end_b
        elif stage["mode"] != "LINEAR":
            end_b = start_b
        command_b.extend([start_b, end_b])
        command_flow.extend([stage["flow_mL_min"]] * 2)
    ymax = max([float(np.max(y)) for _, y in series] + [1e-12]) * 1.05
    cmax = max(float(np.nanmax(conductivity)) if conductivity is not None else 1.0, 1e-12) * 1.05
    fmax = max(float(np.nanmax(command_flow if command_flow else flow_mL_min))
               if command_flow or flow_mL_min is not None else 1.0, 1e-12) * 1.05
    def X(v): return left + float(v) / xmax * pw
    def Y(v): return bottom - float(v) / ymax * ph
    def YB(v): return bottom - float(v) / 100.0 * ph
    def YC(v): return bottom - float(v) / cmax * ph
    def YF(v): return bottom - float(v) / fmax * ph
    axis = '#222'
    palette = ['#222222', '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="max-width:100%;height:auto" role="img" aria-label="Predicted UV chromatogram with programmed percent B, lagged conductivity and flow rate">',
             f'<rect width="{width}" height="{height}" fill="white"/>']
    for i, (name, _) in enumerate(series):
        lx = left + (i % 2) * (pw / 2)
        ly = 24 + (i // 2) * 21
        parts += [f'<line x1="{lx}" y1="{ly-4}" x2="{lx+25}" y2="{ly-4}" stroke="{palette[i%7]}" stroke-width="2"/>',
                  f'<text x="{lx+32}" y="{ly}" font-size="12" fill="{axis}">{html.escape(name)}</text>']
    parts += [f'<line x1="{left}" y1="135" x2="{left+pw}" y2="135" stroke="{axis}"/>',
              f'<text x="{left+pw}" y="111" text-anchor="end" font-size="11" fill="{axis}">Elapsed time (min)</text>']
    stage_colors = {'LOAD': '#d9d9d9', 'PLW': '#cfe8f3', 'ELUTION': '#d8eed8'}
    for stage in stages or []:
        x0, x1 = X(stage['start_CV']), X(stage['end_CV'])
        label = str(stage['marker']) if x1-x0 < 60 else f"{stage['marker']} · {stage['duration_CV']:g} CV"
        parts += [f'<rect x="{x0:.2f}" y="143" width="{x1-x0:.2f}" height="18" fill="{stage_colors.get(stage["kind"], "#eee")}" stroke="#777"/>',
                  f'<text x="{(x0+x1)/2:.2f}" y="156" text-anchor="middle" font-size="10" fill="{axis}">{html.escape(label)}</text>',
                  f'<line x1="{x0:.2f}" y1="161" x2="{x0:.2f}" y2="{bottom}" stroke="#777" stroke-dasharray="4 3"/>']
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{axis}"/>',
              f'<line x1="{left}" y1="{bottom}" x2="{left+pw}" y2="{bottom}" stroke="{axis}"/>']
    for j in range(6):
        xv = xmax * j / 5
        xp = X(xv)
        elapsed = float(np.interp(xv, x, time_min)) if time_min is not None else xv
        parts += [f'<text x="{xp:.2f}" y="131" text-anchor="middle" font-size="10" fill="{axis}">{elapsed:.3g}</text>',
                  f'<text x="{xp:.2f}" y="{bottom+22}" text-anchor="middle" font-size="11" fill="{axis}">{xv:.3g}</text>']
    for j in range(5):
        v = ymax*j/4
        yp = Y(v)
        parts += [f'<line x1="{left}" y1="{yp:.2f}" x2="{left+pw}" y2="{yp:.2f}" stroke="#ddd"/>',
                  f'<text x="{left-10}" y="{yp+4:.2f}" text-anchor="end" font-size="11" fill="{axis}">{v:.4g}</text>']
    def poly(values, ordinate, color, dash='', stroke_width=1.8, sample_cv=None):
        result, points = [], []
        for xv, v in zip(x if sample_cv is None else sample_cv, values):
            if np.isfinite(xv) and np.isfinite(v):
                points.append(f'{X(xv):.2f},{ordinate(v):.2f}')
            elif points:
                result.append(points); points=[]
        if points: result.append(points)
        return ''.join(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-dasharray="{dash}" points="{" ".join(segment)}"/>' for segment in result)
    for i, (_, values) in enumerate(series):
        parts.append(poly(values, Y, palette[i%7], stroke_width=2.2 if i==0 else 1.5, sample_cv=series_x[i]))
    for label, attr, visible, xpos, max_value, ordinate, color in [
        ('Buffer B (%)', 'percent-b', show_percent_B, left+pw, 100.0, YB, '#008c95'),
        ('Conductivity (mS/cm)', 'conductivity', show_conductivity, left+pw+82, cmax, YC, '#b36b00'),
        ('Flow rate (mL/min)', 'flow-rate', show_flow_rate, left+pw+164, fmax, YF, '#6a3d9a'),
    ]:
        parts.append(f'<g data-overlay="{attr}" style="display:{"inline" if visible else "none"}">')
        parts.append(f'<line x1="{xpos}" y1="{top}" x2="{xpos}" y2="{bottom}" stroke="{color}"/>')
        for j in range(5):
            val = max_value*j/4
            yp = ordinate(val)
            parts.append(f'<text x="{xpos+7}" y="{yp+4:.2f}" font-size="10" fill="{color}">{val:.3g}</text>')
        tx = xpos+48
        parts.append(f'<text x="{tx}" y="{(top+bottom)/2}" transform="rotate(-90 {tx} {(top+bottom)/2})" text-anchor="middle" font-size="12" fill="{color}">{label}</text>')
        if attr == 'percent-b':
            if percent_B is not None:
                parts.append(poly(command_b if command_cv else percent_B, YB, color, '2 5', 2.4,
                                  sample_cv=command_cv if command_cv else x))
            if column_percent_B is not None:
                parts.append(poly(column_percent_B, YB, '#50a5aa', '1 4', 1.7))
            parts.append(f'<text x="{left}" y="580" font-size="12" fill="{color}">Dotted: programmed %B; light dotted: column %B</text>')
        elif attr == 'conductivity':
            if conductivity is not None:
                parts.append(poly(conductivity, YC, color, '8 4', 2.1))
            parts.append(f'<text x="{left}" y="601" font-size="12" fill="{color}">Dashed: conductivity at detector (column + conductivity dead volume)</text>')
        else:
            if flow_mL_min is not None:
                parts.append(poly(command_flow if command_cv else flow_mL_min, YF, color, '5 3', 2.1,
                                  sample_cv=command_cv if command_cv else x))
            parts.append(f'<text x="{left}" y="622" font-size="12" fill="{color}">Dashed: programmed stage flow rate</text>')
        parts.append('</g>')
    parts += [f'<text x="{left+pw/2}" y="562" text-anchor="middle" font-size="13" fill="{axis}">Column volumes (CV; run ends at {xmax:g} CV)</text>',
              f'<text x="20" y="{(top+bottom)/2}" transform="rotate(-90 20 {(top+bottom)/2})" text-anchor="middle" font-size="13" fill="{axis}">{html.escape(y_axis_label)}</text>', '</svg>']
    return ''.join(parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def check_result_freshness(
    config: dict[str, Any],
    run_context: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """Reject existing result files unless they match this run and its recipe."""
    root = Path(output_dir) if output_dir is not None else LATEST_RESULTS_MANIFEST.parent
    manifest_path = root / LATEST_RESULTS_MANIFEST.name
    context = run_context or {}
    try:
        if not manifest_path.is_file():
            return False, "No run receipt exists. Run the selected model before opening results."
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("build_id") != RESULT_GUARD_BUILD:
            return False, "The result was produced by another model build. Run the selected model again."
        if manifest.get("configuration_fingerprint") != configuration_fingerprint(config):
            # Least-squares controls do not enter a forward chromatogram.
            # Older receipts can also omit newly added fitting controls, so
            # compare the stored recipe when the full audit hash differs.
            stored = manifest.get("configuration")
            if not isinstance(stored, dict):
                return False, "Current settings differ from the settings used for this chromatogram. Run the selected model again."
            current_output_config = normalize_config(config)
            stored_output_config = normalize_config(stored)
            current_output_config.pop("least_squares", None)
            stored_output_config.pop("least_squares", None)
            if current_output_config != stored_output_config:
                return False, "Current settings differ from the settings used for this chromatogram. Run the selected model again."
        expected_number = int(context.get("run_number", 1))
        expected_name = str(context.get("run_name") or f"Run {expected_number}")
        if int(manifest.get("run_number", -1)) != expected_number or str(manifest.get("run_name")) != expected_name:
            return False, "The chromatogram belongs to a different selected run. Run the selected model again."
        for key in ("html", "csv", "svg"):
            output_path = root / f"latest_results.{key}"
            if not output_path.is_file() or _sha256_file(output_path) != manifest.get("output_sha256", {}).get(key):
                return False, f"The {key.upper()} result changed after the run receipt was written. Run the selected model again."
        return True, f"Result matches Run {expected_number} — {expected_name}; build {RESULT_GUARD_BUILD}."
    except Exception as exc:
        return False, f"Could not verify the result receipt ({type(exc).__name__}: {exc}). Run the selected model again."


def write_outputs(
    config: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path | str | None = None,
    run_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir) if output_dir is not None else LATEST_RESULTS_CSV.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / LATEST_RESULTS_CSV.name
    html_path = output_dir / LATEST_RESULTS_HTML.name
    manifest_path = output_dir / LATEST_RESULTS_MANIFEST.name
    # Invalidate first: a partial or failed write must never leave an old run
    # marked as the current result.
    manifest_path.unlink(missing_ok=True)
    trace = result["trace"]
    names = result["simulation"]["component_names"]
    program: StageProgram = result["simulation"]["program"]
    stages = _stage_receipt(config, program)
    context = dict(run_context or {})
    run_number = int(context.get("run_number", 1))
    run_name = str(context.get("run_name") or f"Run {run_number}")
    fingerprint = configuration_fingerprint(config)
    stage_names = [row["name"] for row in stages]
    affinity_used = result.get("parameter_receipt", {}).get("langmuir_affinity_used", [])
    affinity_fields = [f"{name}_effective_b_L_g" for name in names] if affinity_used else []

    fields = ["run_number", "run_name", "settings_fingerprint", "time_min", "CV", "volume_mL", "uv_mAU",
              "program_stage", "flow_mL_min", "percent_B", "programmed_percent_B", "column_percent_B",
              "programmed_conductivity_mS_cm", "column_conductivity_mS_cm", "column_total_protein_g_L",
              "stage_index", "pH", "salt_concentration_M", "conductivity_mS_cm", "ionic_strength_M",
              "programmed_salt_concentration_M", "binding_salt_concentration_M", "binding_ionic_strength_M",
              "total_protein_g_L"] + [f"{name}_g_L" for name in names] + [f"{name}_uv_mAU" for name in names] + affinity_fields
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for k in range(len(trace["CV"])):
            row: dict[str, Any] = {
                "run_number": run_number, "run_name": run_name, "settings_fingerprint": fingerprint,
                "time_min": trace["time_min"][k], "CV": trace["CV"][k],
                "volume_mL": trace["volume_mL"][k], "uv_mAU": trace["uv_mAU"][k],
                "program_stage": stage_names[program._stage_index_time(float(trace["t_s"][k]))],
                "flow_mL_min": trace["flow_mL_min"][k],
                "percent_B": (float(trace["percent_B"][k]) if math.isfinite(float(trace["percent_B"][k])) else ""),
                "programmed_percent_B": trace["programmed_percent_B"][k],
                "column_percent_B": trace["column_percent_B"][k],
                "programmed_conductivity_mS_cm": trace["programmed_conductivity_mS_cm"][k],
                "column_conductivity_mS_cm": trace["column_conductivity_mS_cm"][k],
                "column_total_protein_g_L": trace["column_total_g_L"][k],
                "total_protein_g_L": trace["total_g_L"][k],
            }
            row.update(stage_index=trace["stage_index"][k], pH=trace["pH"][k],
                       salt_concentration_M=trace["salt_concentration_M"][k],
                       conductivity_mS_cm=trace["conductivity_mS_cm"][k],
                       ionic_strength_M=trace["ionic_strength_M"][k])
            for field in ("programmed_salt_concentration_M", "binding_salt_concentration_M", "binding_ionic_strength_M"):
                row[field] = trace[field][k]
            for i, name in enumerate(names):
                row[f"{name}_g_L"] = trace["component_g_L"][k, i]
                row[f"{name}_uv_mAU"] = trace["component_uv_mAU"][k, i]
                if affinity_used:
                    row[f"{name}_effective_b_L_g"] = trace["component_effective_b_L_g"][k, i]
            writer.writerow(row)

    mb_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [
            name, f"{vals['input_g']:.6g}", f"{vals['outlet_g']:.6g}",
            f"{vals['column_inventory_g']:.6g}", f"{vals['closure_error_percent_of_input']:.4g}%",
        ]) + "</tr>"
        for name, vals in result["mass_balance"].items()
    )
    svg = _svg_plot(
        trace["CV"], [("Total UV", trace["uv_mAU"])] + [(name, trace["component_uv_mAU"][:, i]) for i, name in enumerate(names)],
        stages=stages, percent_B=trace["programmed_percent_B"], column_percent_B=trace["column_percent_B"],
        conductivity=trace["conductivity_mS_cm"], flow_mL_min=trace["flow_mL_min"],
        time_min=trace["time_min"], total_cv=program.total_CV,
        show_percent_B=config["system"]["show_percent_B"],
        show_conductivity=config["system"]["show_conductivity"],
        show_flow_rate=config["system"].get("show_flow_rate", True),
    )
    svg_path = output_dir / "latest_results.svg"
    svg_path.write_text(svg, encoding="utf-8")
    derived = html.escape(json.dumps(derived_input_summary(config), indent=2))
    diagnostics = html.escape(json.dumps(result["simulation"]["diagnostics"], indent=2))
    parameter_receipt = html.escape(json.dumps(result.get("parameter_receipt", {}), indent=2, ensure_ascii=False))
    buffer_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value) if value is not None else '—')}</td>" for value in (
            label, env["salt_input_mode"], env["pH"], env["salt_concentration_M"],
            env["ionic_strength_source"], env["ionic_strength_M"], env["conductivity_mS_cm"],
            env["conductivity_source"],
        )) + "</tr>"
        for label, env in result.get("parameter_receipt", {}).get("buffer_chemistry_used", {}).items()
    )
    stage_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["marker"], row["name"], f'{row["start_CV"]:g}', f'{row["end_CV"]:g}',
            f'{row["duration_CV"]:g}', row["mode"], f'{row["flow_mL_min"]:g}',
            row["chemistry_control"],
            "—" if row["start_percent_B"] is None else f'{float(row["start_percent_B"]):g}% ',
            "—" if row["end_percent_B"] is None else f'{float(row["end_percent_B"]):g}% ',
            f'{row["start_pH"]:g} → {row["end_pH"]:g}',
            f'{row["start_salt_M"]:g} → {row["end_salt_M"]:g}',
            f'{row["start_salt_source"]} ({row["start_salt_input_mode"]}) → {row["end_salt_source"]} ({row["end_salt_input_mode"]})',
            f'{row["start_conductivity_mS_cm"]:g} → {row["end_conductivity_mS_cm"]:g}',
        )) + "</tr>"
        for row in stages
    )
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    diagnostics_obj = result["simulation"]["diagnostics"]
    saturation = diagnostics_obj.get("nominal_langmuir_saturation")
    saturation_html = (
        "<h2>Load versus nominal Langmuir saturation</h2>"
        f"<p>Target feed load: {saturation['target_feed_load_g_L_bed']:.6g} g/L packed bed; "
        f"target stationary-phase ceiling: {saturation['target_nominal_saturation_g_L_bed']:.6g} "
        f"g/L packed bed (qmax {saturation['target_qmax_g_L_stationary_phase']:.6g} "
        f"g/L stationary phase × bed fraction {saturation['stationary_fraction_of_bed']:.6g}); "
        f"load/ceiling: {saturation['target_load_to_saturation_ratio']:.4g}×. "
        "Competition and transport can reduce actual retention below this ceiling.</p>"
    ) if saturation else ""
    affinity_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            component["component"], f'{component["b_L_g"]:.6g}',
            f'{component["salt_sensitivity_per_M"]:.6g}', endpoint["stage"],
            f'{endpoint["start_salt_M"]:.6g} → {endpoint["end_salt_M"]:.6g}',
            f'{endpoint["start_b_eff_L_g"]:.6g} → {endpoint["end_b_eff_L_g"]:.6g}',
        )) + "</tr>"
        for component in affinity_used for endpoint in component["stage_endpoints"]
    )
    affinity_html = (
        "<h2>Competitive Langmuir affinity used</h2>"
        "<p>The solver uses b<sub>eff</sub> = b × exp(k<sub>s</sub> × local binding salt) "
        "in every axial cell. This table evaluates that equation at the programmed stage "
        "endpoints. The CSV records effective b at the column outlet cell on every time "
        "point, using local binding salt before the UV detector delay. An abrupt step to "
        "very low salt may release protein at nearly the same time across a range of "
        "affinities; a gradual salt ramp resolves peak-position sensitivity.</p>"
        "<div class=\"scroll\"><table><thead><tr><th>Component</th><th>b [L/g]</th>"
        "<th>k_s [M⁻¹]</th><th>Stage</th><th>Salt start → end [M]</th>"
        "<th>b_eff start → end [L/g]</th></tr></thead><tbody>"
        + affinity_rows + "</tbody></table></div>"
    ) if affinity_used else ""
    html_path.write_text(f"""<!doctype html>
<meta charset="utf-8"><title>Chromatography model results</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:Arial,sans-serif;margin:2rem;max-width:1100px}}svg{{display:block;width:100%;max-width:1000px;height:auto;background:#fff}}table{{border-collapse:collapse;max-width:100%;font-size:.9rem}}td,th{{border:1px solid #bbb;padding:.35rem .55rem;text-align:right}}td:first-child,th:first-child{{text-align:left}}pre{{background:#f4f4f4;padding:1rem;overflow:auto}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>Chromatography model results</h1>
<p><b>Run:</b> {run_number} — {html.escape(run_name)} &nbsp; <b>Build:</b> {RESULT_GUARD_BUILD} &nbsp; <b>Generated UTC:</b> {generated_utc}</p>
<p><b>Model:</b> {html.escape(MODEL_LABELS[config['model']])}</p>
<p><b>Settings fingerprint:</b> <code>{fingerprint}</code></p>
<p><label><input type="checkbox" {'checked' if config['system']['show_percent_B'] else ''} onchange="document.querySelectorAll('[data-overlay=percent-b]').forEach(el=>el.style.display=this.checked?'':'none')"> Show %B</label>
&nbsp; <label><input type="checkbox" {'checked' if config['system']['show_conductivity'] else ''} onchange="document.querySelectorAll('[data-overlay=conductivity]').forEach(el=>el.style.display=this.checked?'':'none')"> Show conductivity</label>
&nbsp; <label><input type="checkbox" {'checked' if config['system'].get('show_flow_rate', True) else ''} onchange="document.querySelectorAll('[data-overlay=flow-rate]').forEach(el=>el.style.display=this.checked?'':'none')"> Show flow rate</label></p>
{svg}
<p>{html.escape(' '.join(result['simulation']['diagnostics'].get('warnings', [])))}</p>
<h2>Resolved buffer chemistry</h2>
<p>Buffer references and %B stages use these current definitions. Direct endpoint stages use the recipe table below. HIC binding uses salt molarity; native CPA ion exchange uses ionic strength. The conversion factor is {config['conversion']['conductivity_to_salt_M_per_mS_cm']:g} M/(mS/cm). SALT_FACTOR_ESTIMATE marks an estimated conductivity overlay.</p>
<div class="scroll"><table><thead><tr><th>Buffer</th><th>Salt input</th><th>pH</th><th>Salt M</th><th>Ionic source</th><th>Ionic strength M</th><th>Conductivity mS/cm</th><th>Conductivity source</th></tr></thead><tbody>{buffer_rows}</tbody></table></div>
<h2>Process recipe used for this chromatogram</h2>
<p>Total run: {program.total_CV:g} CV; {len(trace['CV']) - 1} reported time steps; {diagnostics_obj.get('axial_cells', '—')} axial cells. Marker numbers in the chart match this table.</p>
<p>The chart outputs UV in mAU, with dotted programmed %B and lagged column %B, conductivity, and the programmed stage flow rate on their own axes in the same plot. The three overlays can be toggled independently. CSV also records programmed salt, column salt and the outlet cell's binding salt / ionic strength (pore phase for TDM). Detector dead volumes are independent downstream delays; the column mass balance uses the undelayed outlet. %B is undefined for direct endpoint chemistry stages. pH blending is a linear process approximation, without acid-base speciation.</p>
<div class="scroll"><table><thead><tr><th>Marker</th><th>Stage</th><th>Start CV</th><th>End CV</th><th>Duration CV</th><th>Mode</th><th>Flow mL/min</th><th>Chemistry control</th><th>Start %B</th><th>End %B</th><th>pH start → end</th><th>Salt M start → end</th><th>Salt source/input start → end</th><th>Conductivity mS/cm start → end</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
{affinity_html}
{saturation_html}
<h2>Mass balance</h2>
<table><thead><tr><th>Component</th><th>Input g</th><th>Outlet g</th><th>Column inventory g</th><th>Closure error</th></tr></thead><tbody>{mb_rows}</tbody></table>
<h2>Inputs actually used by this run</h2><pre>{parameter_receipt}</pre>
<h2>Derived values</h2><pre>{derived}</pre>
<h2>Solver diagnostics</h2><pre>{diagnostics}</pre>
<p>Point data: <code>{html.escape(csv_path.name)}</code></p>
""", encoding="utf-8")
    manifest = {
        "schema": 1,
        "build_id": RESULT_GUARD_BUILD,
        "generated_utc": generated_utc,
        "run_number": run_number,
        "run_name": run_name,
        "configuration_fingerprint": fingerprint,
        "configuration": normalize_config(config),
        "total_CV": float(program.total_CV),
        "load_capacity_g_L_resin": float(config["feed"]["load_density_mg_mL_resin"]),
        "required_load_volume_CV": float(program.stages[0].duration_CV),
        "required_load_volume_mL": float(program.stages[0].duration_CV) * float(config["column"]["volume_mL"]),
        "time_steps": len(trace["CV"]) - 1,
        "axial_cells": diagnostics_obj.get("axial_cells"),
        "stages": stages,
        "warnings": diagnostics_obj.get("warnings", []),
        "nominal_langmuir_saturation": saturation,
        "langmuir_affinity_used": affinity_used,
        "target_peak": result.get("parameter_receipt", {}).get("target_peak"),
        "output_sha256": {
            "html": _sha256_file(html_path), "csv": _sha256_file(csv_path), "svg": _sha256_file(svg_path),
        },
    }
    _write_manifest(manifest_path, manifest)
    return {"csv": str(csv_path), "html": str(html_path), "svg": str(svg_path), "manifest": str(manifest_path)}


def run_and_write(
    config: dict[str, Any],
    output_dir: Path | str | None = None,
    run_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    result = simulate(config)
    return write_outputs(config, result, output_dir, run_context=run_context)

def run_batch(
    base_config: dict[str, Any],
    run_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run several operating-condition variants with one locked parameter set.

    `base_config` is the selected batch's mechanistic parameter set.  Each item
    in `run_configs` may vary run conditions, but any attempted change to the
    selected model, column identity, transport/isotherm parameters, impurity
    selection, species composition or species parameters is overwritten by the
    base batch values before simulation.
    """
    if not run_configs:
        raise ValueError("A batch must contain at least one run.")
    expected_fingerprint = batch_parameter_fingerprint(base_config)
    results: list[dict[str, Any]] = []
    for index, requested in enumerate(run_configs, start=1):
        effective, attempted_overrides = apply_batch_shared_parameters(base_config, requested)
        errors = validate_config(effective, strict=True)
        if errors:
            raise ValueError(
                f"Batch run {index} validation failed:\n- " + "\n- ".join(errors)
            )
        actual_fingerprint = batch_parameter_fingerprint(effective)
        if actual_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"Batch run {index} did not inherit the selected shared parameter set."
            )
        result = simulate(effective)
        result["batch"] = {
            "run_number": index,
            "shared_parameter_fingerprint": expected_fingerprint,
            "shared_parameters": batch_shared_parameter_snapshot(base_config),
            "attempted_shared_overrides_ignored": attempted_overrides,
        }
        results.append({"config": effective, "result": result})
    return results


def run_batch_and_write(
    base_config: dict[str, Any],
    run_configs: list[dict[str, Any]],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Execute a batch and save a simple stable gallery for up to 30 runs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    batch_results = run_batch(base_config, run_configs)
    rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for raw_run, item in zip(run_configs, batch_results):
        run_number = int(item["result"]["batch"]["run_number"])
        run_name = str(raw_run.get("_run_name") or raw_run.get("run_name") or f"Run {run_number}")
        run_dir = root / f"run_{run_number:03d}"
        paths = write_outputs(item["config"], item["result"], run_dir, run_context={"run_number": run_number, "run_name": run_name})
        rows.append({
            "run_number": run_number,
            "run_name": run_name,
            "shared_parameter_fingerprint": item["result"]["batch"]["shared_parameter_fingerprint"],
            "ignored_shared_override_count": len(item["result"]["batch"]["attempted_shared_overrides_ignored"]),
            "html": paths["html"],
            "csv": paths["csv"],
            "manifest": paths["manifest"],
        })
        rel_html = Path(paths["html"]).relative_to(root).as_posix()
        rel_svg = Path(paths["svg"]).relative_to(root).as_posix()
        cards.append(f'<section><h2>{html.escape(run_name)}</h2><img src="{html.escape(rel_svg)}" style="display:block;width:100%;max-width:1000px;height:auto;background:#fff"><p><a href="{html.escape(rel_html)}">Open detailed result</a></p></section>')
    summary_path = root / "batch_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    gallery_path = root / "batch_gallery.html"
    gallery_path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Chromatography batch gallery</title>'
        '<style>body{font-family:Arial,sans-serif;margin:2rem;max-width:1150px}section{border-bottom:1px solid #bbb;padding:1rem 0}img{background:#fff}</style>'
        '<h1>Chromatography batch gallery</h1>' + ''.join(cards), encoding="utf-8")
    return {
        "summary_csv": str(summary_path),
        "gallery_html": str(gallery_path),
        "shared_parameter_fingerprint": batch_parameter_fingerprint(base_config),
        "runs": rows,
    }
