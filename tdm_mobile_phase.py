"""Protein-independent buffer transport, shared by all column models.

The buffer mixer is a stirred volume upstream of the column. Column salt is
unretained, dispersive, and fully pore-accessible in TDM. Detector dead volumes
are applied separately, downstream, in accumulated volume rather than time.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix


class MobilePhaseTransport:
    def __init__(self, config, program, scalar_transport):
        self.program = program
        self.config = config
        # Transport analytical salt independently: conductivity is a calibrated
        # proxy only when the user selected it as the chemistry input.
        fields = 5  # pH, salt molarity, ionic strength, conductivity, %B
        self.nx = int(config['numerics']['axial_positions'])
        self.tdm = config['model'].startswith('TDM')
        length = float(config['column']['length_mm']) * 1e-3
        volume = float(config['column']['volume_mL']) * 1e-6
        area = volume / length
        dx = length / self.nx
        self.eps = float(config['tdm']['void_fraction'] if self.tdm else config['edm']['total_bed_porosity'])
        epsp = float(config['tdm']['particle_porosity']) if self.tdm else 0.0
        system = config['system']
        mixer_m3 = float(system['buffer_dispersion_mL']) * 1e-6 if system['buffer_dispersion_enabled'] else 0.0
        dispersion = float(system['salt_dispersion_mm2_s']) * 1e-6 if system['salt_dispersion_enabled'] else 0.0
        self.delay_cv = self.eps + (1.0 - self.eps) * epsp

        def inlet(t):
            cv = program.time_to_cv(t)
            pH, salt, ionic, cond, _stage = program.chemistry_at_cv(cv)
            pct = program.percent_B_at_cv(cv)
            # Undefined %B in endpoint-chemistry stages is handled as a display
            # mask. Salt, ionic strength and pH always use the actual endpoints.
            return np.array([pH, salt, ionic, cond, pct if np.isfinite(pct) else 0.0])

        initial = inlet(0.0)
        block = self.nx * fields
        initial_bulk = np.tile(initial, (self.nx, 1)).ravel()
        y0 = np.concatenate([initial, initial_bulk, initial_bulk]) if self.tdm else np.concatenate([initial, initial_bulk])

        def rhs(t, y):
            target = inlet(t)
            flow = program.flow_at_time(t) * 1e-6 / 60.0
            velocity = flow / (area * self.eps)
            if mixer_m3 > 0:
                mixed = y[:fields]
                dmix = flow / mixer_m3 * (target - mixed)
            else:
                mixed = target
                dmix = np.zeros(fields)
            bulk = y[fields:fields + block].reshape(self.nx, fields)
            dbulk = np.column_stack([
                scalar_transport(bulk[:, i], mixed[i], velocity, dispersion, dx) for i in range(fields)
            ])
            if self.tdm:
                pore = y[fields + block:].reshape(self.nx, fields)
                # k_eff,salt = r_p/3 gives 3*k_eff/r_p = 1 s^-1,
                # the negligible-resistance assumption in the supplied CPA study.
                transfer = bulk - pore
                dbulk -= (1.0 - self.eps) / self.eps * transfer
                dpore = transfer / epsp
                return np.concatenate([dmix, dbulk.ravel(), dpore.ravel()])
            return np.concatenate([dmix, dbulk.ravel()])

        # Independent fields; limited second-order fluxes span two neighbors.
        # Sparse differences keep full-resolution TDM chemistry inexpensive.
        sparsity = lil_matrix((len(y0), len(y0)), dtype=int)
        for field in range(fields):
            sparsity[field, field] = 1
            for cell in range(self.nx):
                row = fields + cell * fields + field
                for neighbor in range(max(0, cell - 2), min(self.nx, cell + 3)):
                    sparsity[row, fields + neighbor * fields + field] = 1
                if cell == 0:
                    sparsity[row, field] = 1
                if self.tdm:
                    pore_row = row + block
                    sparsity[row, pore_row] = 1
                    sparsity[pore_row, row] = 1
                    sparsity[pore_row, pore_row] = 1
        self.segments = []
        start = 0.0
        # Restart at every recipe boundary so short load/wash steps cannot be
        # skipped by an adaptive integration step.
        for end in program.stage_end_time_s:
            if end <= start:
                continue
            end_inside = np.nextafter(end, start)
            def stage_rhs(t, y, stop=end_inside):
                return rhs(min(t, stop), y)
            solution = solve_ivp(stage_rhs, (start, end), y0, method='BDF',
                                 rtol=2e-7, atol=1e-9, dense_output=True,
                                 jac_sparsity=sparsity.tocsr(),
                                 max_step=max((end - start) / 25.0, 1e-6))
            if not solution.success:
                raise RuntimeError('Buffer/salt transport failed: ' + solution.message)
            self.segments.append((end, solution.sol))
            y0 = solution.y[:, -1]
            start = end
        self._cache_time = None

    def _state(self, t):
        t = float(np.clip(t, 0.0, self.program.total_time_s))
        if self._cache_time != t:
            for end, solution in self.segments:
                if t <= end + 1e-12:
                    self._cache_state = solution(t)
                    break
            self._cache_time = t
        return self._cache_state

    def local(self, t, *, pore=False):
        y = self._state(t)
        start = 5 + (self.nx * 5 if pore and self.tdm else 0)
        raw = y[start:start + self.nx * 5].reshape(self.nx, 5)
        # Public order: pH, salt molarity, ionic strength, conductivity, %B.
        return np.column_stack([raw[:, 0], np.maximum(raw[:, 1], 0),
                                np.maximum(raw[:, 2], 0), np.maximum(raw[:, 3], 0), raw[:, 4]])

    def outlet(self, times):
        values = np.asarray([self.local(t)[-1] for t in times])
        for i, t in enumerate(times):
            source_cv = max(self.program.time_to_cv(float(t)) - self.delay_cv, 0.0)
            if not np.isfinite(self.program.percent_B_at_cv(source_cv)):
                values[i, 4] = np.nan
        return values


def delay_by_volume(cv, values, dead_volume_cv, *, initial=0.0):
    """Plug-flow detector line delay, valid across changes in recipe flow."""
    values = np.asarray(values, dtype=float)
    if dead_volume_cv <= 0:
        return values.copy()
    source = np.asarray(cv, dtype=float) - float(dead_volume_cv)
    if values.ndim == 1:
        return np.interp(source, cv, values, left=float(initial))
    return np.column_stack([np.interp(source, cv, values[:, i], left=float(initial))
                            for i in range(values.shape[1])])
