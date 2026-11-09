"""Search a chosen multi-stage method using the unchanged production simulator."""
from __future__ import annotations

import copy
import csv
import html
import json
import math
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from tdm_method_optimisation import _pool_masses, _safe_name
from tdm_minimal_io import (MAX_ELUTION_STEPS, MAX_PLW_STEPS, active_components,
                            batch_parameter_fingerprint, configuration_fingerprint,
                            normalize_config, validate_config)
from tdm_minimal_model import _svg_plot, simulate


OBJECTIVES = {
    "RECOVERY": ("yield_percent", "Maximise product recovery"),
    "PURITY": ("purity_percent", "Maximise product purity"),
    "PRODUCTIVITY": ("productivity_g_L_resin_h", "Maximise product mass per resin volume per modelled cycle hour"),
    "PRODUCT_MASS": ("product_mass_g", "Maximise product mass per batch"),
    "POOL_CONCENTRATION": ("pool_concentration_g_L", "Maximise product pool concentration"),
    "BUFFER_EFFICIENCY": ("buffer_efficiency_g_L", "Maximise product mass per wash and elution buffer litre"),
    "SHORT_CYCLE": ("negative_run_time_min", "Minimise modelled load, wash and elution cycle time"),
    "PEAK_SEPARATION": ("minimum_peak_resolution", "Maximise worst product-to-impurity peak resolution"),
}
STYLES = ("GRADIENT", "ISOCRATIC", "BOTH")


def variable_names(washes: int, elutions: int, style: str) -> tuple[str, ...]:
    names = ["loading_g_L", "flow_mL_min"]
    for i in range(1, washes + 1):
        names += [f"plw{i}_CV", f"plw{i}_B"]
    for i in range(1, elutions + 1):
        names.append(f"el{i}_CV")
        if style == "GRADIENT" or (style == "BOTH" and i == 1):
            names += [f"el{i}_start_B", f"el{i}_end_B"]
        else:
            names.append(f"el{i}_B")
    names += ["pool_start_CV", "pool_end_CV"]
    return tuple(names)


def _recipe(config: dict[str, Any], values: dict[str, float], washes: int,
            elutions: int, style: str) -> dict[str, Any]:
    out = copy.deepcopy(config)
    p = out["process"]
    p["plw_count"], p["elution_count"] = washes, elutions
    out["feed"]["load_density_mg_mL_resin"] = values["loading_g_L"]
    p["load_amount_basis"] = "CAPACITY"
    p["load_CV"] = values["loading_g_L"] / float(out["feed"]["total_concentration_mg_mL"])
    p["load_volume_mL"] = p["load_CV"] * float(out["column"]["volume_mL"])
    out["column"]["flow_mL_min"] = values["flow_mL_min"]
    for i in range(1, washes + 1):
        step = p["plw_steps"][i-1]
        step.update(mode="SET_POINT", chemistry_control="BUFFER_B_PERCENT",
                    CV=values[f"plw{i}_CV"], flow_mL_min=values["flow_mL_min"],
                    start_percent_B=values[f"plw{i}_B"], end_percent_B=values[f"plw{i}_B"])
    for i in range(1, elutions + 1):
        step = p["elution_steps"][i-1]
        gradient = style == "GRADIENT" or (style == "BOTH" and i == 1)
        start = values[f"el{i}_start_B"] if gradient else values[f"el{i}_B"]
        end = values[f"el{i}_end_B"] if gradient else start
        step.update(mode="LINEAR" if gradient else "SET_POINT",
                    chemistry_control="BUFFER_B_PERCENT", CV=values[f"el{i}_CV"],
                    flow_mL_min=values["flow_mL_min"], start_percent_B=start,
                    end_percent_B=end)
    return out


def _peak_resolution(cv: np.ndarray, concentrations: np.ndarray,
                     elution_start: float) -> tuple[float, list[dict[str, Any]]]:
    """Return minimum R_s = 1.18 ΔCV/(FWHM_1 + FWHM_2), or zero if unresolved."""
    mask = cv >= elution_start
    x, y = cv[mask], np.maximum(concentrations[mask], 0)
    peaks: list[tuple[float, float] | None] = []
    for column in y.T:
        if len(x) < 3 or not np.isfinite(column).all() or column.max() <= 0:
            peaks.append(None)
            continue
        k = int(np.argmax(column))
        half = column[k] / 2
        left = k
        while left > 0 and column[left] > half:
            left -= 1
        right = k
        while right < len(x)-1 and column[right] > half:
            right += 1
        if left == 0 and column[left] > half or right == len(x)-1 and column[right] > half:
            peaks.append(None)
            continue
        xl = float(np.interp(half, [column[left], column[left+1]], [x[left], x[left+1]]))
        xr = float(np.interp(half, [column[right], column[right-1]], [x[right], x[right-1]]))
        peaks.append((float(x[k]), max(xr-xl, 1e-9)))
    detail = []
    for i in range(1, len(peaks)):
        product, impurity = peaks[0], peaks[i]
        resolution = (1.18 * abs(product[0]-impurity[0]) / (product[1]+impurity[1])
                      if product is not None and impurity is not None else 0.0)
        detail.append({"impurity_index": i, "resolution": float(resolution),
                       "product_peak_CV": product[0] if product else None,
                       "impurity_peak_CV": impurity[0] if impurity else None})
    return (min((row["resolution"] for row in detail), default=0.0), detail)


def _metrics(result: dict[str, Any], config: dict[str, Any], values: dict[str, float],
             options: dict[str, Any]) -> dict[str, Any]:
    program = result["simulation"]["program"]
    elution_start = float(program.stage_start_CV[1 + int(config["process"]["plw_count"])])
    start, end = elution_start + values["pool_start_CV"], elution_start + values["pool_end_CV"]
    trace = result["trace"]
    masses = _pool_masses(trace, start, end, float(config["column"]["volume_mL"]))
    product, total = float(masses[0]), float(masses.sum())
    names = [str(s["name"]) for s in active_components(config)]
    fractions = {name: 100 * float(masses[i]) / total if total else 0.0
                 for i, name in enumerate(names)}
    loaded = (float(config["feed"]["load_density_mg_mL_resin"])
              * float(config["column"]["volume_mL"]) / 1000
              * float(active_components(config)[0]["mass_percent"]) / 100)
    volume = (end-start) * float(config["column"]["volume_mL"])
    run_time = float(program.total_time_s) / 60
    buffer_volume_L = (sum(float(s["CV"]) for key, count in
                           (("plw_steps", "plw_count"), ("elution_steps", "elution_count"))
                           for s in config["process"][key][:int(config["process"][count])])
                       * float(config["column"]["volume_mL"]) / 1000)
    resolution, peak_details = _peak_resolution(
        np.asarray(trace["CV"], dtype=float), np.asarray(trace["column_component_g_L"], dtype=float),
        elution_start)
    for row in peak_details:
        row["impurity"] = names[row.pop("impurity_index")]
    pressure = (float(options["reference_pressure_bar"]) * values["flow_mL_min"]
                / float(options["reference_flow_mL_min"])) if options.get("max_pressure_bar") is not None else None
    purity, recovery = fractions[names[0]], 100*product/loaded if loaded > 0 else 0.0
    margins = {
        "minimum_product_purity_percent": purity - float(options["min_purity_percent"]),
        "minimum_product_recovery_percent": recovery - float(options.get("min_recovery_percent") or 0),
        "minimum_product_mass_g": product - float(options.get("min_product_mass_g") or 0),
        "maximum_pool_volume_mL": float(options["max_pool_volume_mL"]) - volume,
        "pool_order_CV": values["pool_end_CV"]-values["pool_start_CV"]-1e-4,
        "pool_within_recipe_CV": sum(float(s["CV"]) for s in
                                     config["process"]["elution_steps"][:int(config["process"]["elution_count"])])
                                 - values["pool_end_CV"],
    }
    for key, metric in (("max_run_time_min", run_time), ("max_buffer_volume_L", buffer_volume_L)):
        if options.get(key) is not None:
            margins[key] = float(options[key]) - metric
    for name, cap in options.get("max_impurity_percent", {}).items():
        margins[f"maximum_{name}_percent"] = float(cap)-fractions[name]
    if pressure is not None:
        margins["maximum_pressure_bar"] = float(options["max_pressure_bar"])-pressure
    return {"yield_percent": recovery, "purity_percent": purity,
            "product_mass_g": product, "pool_concentration_g_L": product*1000/volume,
            "productivity_g_L_resin_h": product / (float(config["column"]["volume_mL"])/1000 * run_time/60),
            "buffer_efficiency_g_L": product/buffer_volume_L if buffer_volume_L else 0.0,
            "buffer_volume_L": buffer_volume_L, "run_time_min": run_time,
            "negative_run_time_min": -run_time, "minimum_peak_resolution": resolution,
            "peak_resolutions": peak_details, "pool_volume_mL": volume,
            "pool_start_CV": start, "pool_end_CV": end,
            "pool_start_min": float(program.cv_to_time(start))/60,
            "pool_end_min": float(program.cv_to_time(end))/60,
            "loaded_product_g": loaded, "masses_g": masses.tolist(),
            "fractions_percent": fractions, "estimated_pressure_bar": pressure,
            "margins": margins, "feasible": all(v >= -1e-6 for v in margins.values())}


def _optimise_pool_window(result: dict[str, Any], config: dict[str, Any],
                          values: dict[str, float], options: dict[str, Any],
                          start_bounds: tuple[float, float], end_bounds: tuple[float, float],
                          metric_key: str, grid_size: int = 17
                          ) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Refine collection cuts on one simulated profile, without another model solve.

    Grid screening supplies feasible starts; continuous cut refinement follows
    the fixed-chromatogram strategy in Leweke & von Lieres (2018), Sec. 6.2.2.
    """
    program = result["simulation"]["program"]
    elution_start = float(program.stage_start_CV[1 + int(config["process"]["plw_count"])])
    available = min(sum(float(s["CV"]) for s in config["process"]["elution_steps"][:
                        int(config["process"]["elution_count"])]),
                    float(result["trace"]["CV"][-1]) - elution_start)
    max_span = float(options["max_pool_volume_mL"]) / float(config["column"]["volume_mL"])
    start_lo, start_hi = start_bounds[0], min(start_bounds[1], available - 1.01e-4)
    end_lo, end_hi = end_bounds[0], min(end_bounds[1], available)
    if start_lo > start_hi or end_lo > end_hi:
        return None

    def measure(x: np.ndarray) -> dict[str, Any] | None:
        start, end = map(float, x)
        if (end - start <= 1e-4 or end - start > max_span + 1e-9 or
                start < start_lo - 1e-9 or start > start_hi + 1e-9 or
                end < end_lo - 1e-9 or end > end_hi + 1e-9):
            return None
        try:
            return _metrics(result, config, dict(values, pool_start_CV=start,
                                                 pool_end_CV=end), options)
        except (ValueError, FloatingPointError):
            return None

    pairs = [np.array([values["pool_start_CV"], values["pool_end_CV"]], dtype=float)]
    for start in np.linspace(start_lo, start_hi, grid_size):
        pairs.extend(np.array([start, end]) for end in np.linspace(end_lo, end_hi, grid_size))
        pairs.append(np.array([start, min(end_hi, start + max_span)]))
    measured = [(x, m) for x in pairs if (m := measure(x)) is not None]
    if not measured:
        return None
    feasible = [(x, m) for x, m in measured if m["feasible"]]
    rank_key = "yield_percent" if metric_key == "minimum_peak_resolution" else metric_key
    feasible.sort(key=lambda item: (item[1][rank_key], item[1]["yield_percent"]), reverse=True)
    best_x, best_m = feasible[0] if feasible else (None, None)

    def objective(x: np.ndarray) -> float:
        m = measure(x)
        return -float(m[rank_key]) if m is not None else 1e6

    def margin(x: np.ndarray, name: str) -> float:
        m = measure(x)
        return float(m["margins"][name]) if m is not None else -1e3

    near_feasible = (sorted(
        measured, key=lambda item: sum(max(0.0, -float(margin))**2
                                       for margin in item[1]["margins"].values()))[:3]
                     if not feasible else [])
    for seed, seed_metrics in feasible[:3] + near_feasible:
        trial = minimize(objective, seed, method="SLSQP",
                         bounds=[(start_lo, start_hi), (end_lo, end_hi)],
                         constraints=[{"type": "ineq", "fun": margin, "args": (name,)}
                                      for name in seed_metrics["margins"]],
                         options={"maxiter": 60, "ftol": 1e-9, "eps": 1e-6})
        refined = measure(trial.x)
        if (refined is not None and refined["feasible"] and
                (best_m is None or refined[rank_key] > best_m[rank_key] + 1e-10 or
                 (abs(refined[rank_key] - best_m[rank_key]) <= 1e-10 and
                  refined["yield_percent"] > best_m["yield_percent"] + 1e-10))):
            best_x, best_m = trial.x, refined
    return (dict(values, pool_start_CV=float(best_x[0]), pool_end_CV=float(best_x[1])), best_m) if best_m is not None else None


def _validate(config: dict[str, Any], options: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    errors = validate_config(config, strict=True)
    if errors:
        raise ValueError("Complete and validate the selected run: " + "; ".join(errors))
    names = [str(x["name"]).strip() for x in active_components(config)]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("Product and impurity names must be distinct and nonblank.")
    washes, elutions, style = int(options["plw_count"]), int(options["elution_count"]), str(options["elution_style"])
    if not 0 <= washes <= MAX_PLW_STEPS or not 1 <= elutions <= MAX_ELUTION_STEPS or style not in STYLES:
        raise ValueError("Choose 0–5 washes, 1–5 elutions and gradient, isocratic or both.")
    if style == "BOTH" and elutions < 2:
        raise ValueError("Gradient plus isocratic requires at least two elution steps.")
    if options.get("objective") not in OBJECTIVES:
        raise ValueError("Choose a supported method objective.")
    if options["objective"] == "PEAK_SEPARATION" and len(names) < 2:
        raise ValueError("Peak separation needs at least one impurity in the model.")
    variables = variable_names(washes, elutions, style)
    try:
        lower = np.array([float(options["bounds"][k][0]) for k in variables])
        upper = np.array([float(options["bounds"][k][1]) for k in variables])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Enter lower and upper bounds for every selected method step.") from exc
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.all(lower < upper):
        raise ValueError("All method bounds must be finite, with minimum less than maximum.")
    for i, key in enumerate(variables):
        if key.endswith("_B") and not (0 <= lower[i] < upper[i] <= 100):
            raise ValueError(f"{key} must lie between 0 and 100% B.")
        if key.endswith("_CV") and (lower[i] < 0 if key == "pool_start_CV" else lower[i] <= 0):
            raise ValueError(f"{key} must have a nonnegative pool start or positive stage duration.")
        if key in ("loading_g_L", "flow_mL_min") and lower[i] <= 0:
            raise ValueError(f"{key} must be positive.")
    if lower[-1] <= lower[-2]:
        raise ValueError("Pool end bounds must permit a window after pool start.")
    if options.get("min_purity_percent") is None or options.get("max_pool_volume_mL") is None:
        raise ValueError("Enter minimum purity and maximum pool volume.")
    if (lower[-1] > sum(upper[variables.index(f"el{i}_CV")] for i in range(1, elutions+1))
            or float(options["max_pool_volume_mL"]) / float(config["column"]["volume_mL"]) <= 1e-4):
        raise ValueError("Pool bounds and volume limit leave no collection window in the elution program.")
    for key in ("min_purity_percent", "min_recovery_percent", "min_product_mass_g",
                "max_pool_volume_mL", "max_run_time_min", "max_buffer_volume_L"):
        if options.get(key) is None:
            continue
        val = float(options[key])
        if not math.isfinite(val) or val < 0 or (key.startswith("max_") and val <= 0) or (key.endswith("percent") and val > 100):
            raise ValueError(f"Invalid {key} constraint.")
    for name, cap in options.get("max_impurity_percent", {}).items():
        if name not in names[1:] or not math.isfinite(float(cap)) or not 0 <= float(cap) <= 100:
            raise ValueError("Impurity caps must name a model impurity and be 0–100%.")
    pressure_keys = ("max_pressure_bar", "reference_pressure_bar", "reference_flow_mL_min")
    if any(options.get(k) is not None for k in pressure_keys):
        if any(options.get(k) is None or not math.isfinite(float(options[k])) or float(options[k]) <= 0 for k in pressure_keys):
            raise ValueError("Pressure constraint needs a positive cap, measured reference pressure and reference flow.")
        if float(options["reference_pressure_bar"])*lower[1]/float(options["reference_flow_mL_min"]) > float(options["max_pressure_bar"]):
            raise ValueError("Every permitted flow exceeds the pressure cap estimate.")
    cap = float(options.get("max_evaluations", 30))
    timeout = float(options.get("max_trial_minutes", 0))
    if not math.isfinite(cap) or cap != int(cap) or not 2 <= cap <= 1000:
        raise ValueError("Evaluation cap must be an integer from 2 to 1000.")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Trial timeout must be nonnegative.")
    return variables, lower, upper


def optimise_method(config: dict[str, Any], options: dict[str, Any], *, run_name: str = "Run",
                    output_root: Path | str = "Method_Optimisation",
                    simulator: Callable[[dict[str, Any]], dict[str, Any]] = simulate) -> dict[str, Any]:
    """Search fixed user-selected structure and return best observed feasible candidate."""
    config = normalize_config(config)
    variables, lower, upper = _validate(config, options)
    washes, elutions, style = int(options["plw_count"]), int(options["elution_count"]), options["elution_style"]
    metric_key = OBJECTIVES[options["objective"]][0]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"{_safe_name(run_name)}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir()
    history = folder / "evaluation_history.csv"
    fields = ["candidate", "model_evaluation", "simulation_reused", "phase", *variables,
              "objective_value", "yield_percent", "purity_percent", "product_mass_g",
              "productivity_g_L_resin_h", "pool_concentration_g_L", "buffer_efficiency_g_L",
              "minimum_peak_resolution", "run_time_min", "feasible", "error", "margins_json"]
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    simulations: OrderedDict[tuple[float, ...], tuple[int, dict[str, Any] | None, str | None]] = OrderedDict()
    best: dict[str, Any] | None = None
    solve_count = 0
    reused_count = 0
    cap = int(options.get("max_evaluations", 30))
    bed_mL = float(config["column"]["volume_mL"])
    pool_limit_CV = float(options["max_pool_volume_mL"])/bed_mL
    ix = {k: i for i, k in enumerate(variables)}
    nonpool = variables[:-2]

    def better_metrics(candidate: dict[str, Any], previous: dict[str, Any] | None) -> bool:
        if previous is None:
            return True
        difference = float(candidate[metric_key]) - float(previous[metric_key])
        return difference > 1e-10 or (abs(difference) <= 1e-10 and
                                     candidate["yield_percent"] > previous["yield_percent"])

    def valid_geometry(v: dict[str, float]) -> bool:
        return (v["pool_end_CV"]-v["pool_start_CV"] > 1e-4 and
                v["pool_end_CV"]-v["pool_start_CV"] <= pool_limit_CV+1e-9 and
                v["pool_end_CV"] <= sum(v[f"el{i}_CV"] for i in range(1, elutions+1))+1e-9)

    def repair(u: np.ndarray) -> np.ndarray:
        v = lower + np.clip(u, 0, 1)*(upper-lower)
        if options.get("max_pressure_bar") is not None:
            v[ix["flow_mL_min"]] = min(v[ix["flow_mL_min"]],
                float(options["max_pressure_bar"])*float(options["reference_flow_mL_min"])/float(options["reference_pressure_bar"]))
        available = sum(v[ix[f"el{i}_CV"]] for i in range(1, elutions+1))
        s, e = ix["pool_start_CV"], ix["pool_end_CV"]
        lo = max(lower[s], lower[e]-pool_limit_CV)
        hi = min(upper[s], available-1.01e-4, upper[e]-1.01e-4)
        if lo <= hi:
            v[s] = np.clip(v[s], lo, hi)
            lo_e = max(lower[e], v[s]+1.01e-4)
            hi_e = min(upper[e], available, v[s]+pool_limit_CV)
            if lo_e <= hi_e:
                v[e] = np.clip(v[e], lo_e, hi_e)
        return (v-lower)/(upper-lower)

    with history.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        def add(entry: dict[str, Any], eval_id: int | None, reused: bool) -> None:
            cache[tuple(entry["values"][k] for k in variables)] = entry
            m = entry["metrics"] or {}
            writer.writerow({"candidate": len(cache), "model_evaluation": eval_id or "",
                             "simulation_reused": reused, "phase": entry["phase"], **entry["values"],
                             "objective_value": m.get(metric_key, ""),
                             **{k: m.get(k, "") for k in fields if k in m},
                             "feasible": m.get("feasible", False), "error": entry["error"] or "",
                             "margins_json": json.dumps(m.get("margins", {}))})
            fh.flush()

        def consider(entry: dict[str, Any], result: dict[str, Any]) -> None:
            nonlocal best
            m = entry["metrics"]
            if m is not None and m["feasible"] and better_metrics(m, best["metrics"] if best else None):
                if best is not None:
                    best["result"] = None
                best = entry
                entry["result"] = result

        def evaluate(u: np.ndarray, phase: str) -> dict[str, Any]:
            nonlocal solve_count, reused_count
            v = dict(zip(variables, (lower+np.clip(u, 0, 1)*(upper-lower)).tolist()))
            key = tuple(v[k] for k in variables)
            if key in cache:
                return cache[key]
            entry = {"values": v, "metrics": None, "error": None, "result": None, "phase": phase}
            eval_id = None
            reused = False
            new = False
            result = None
            try:
                if not valid_geometry(v):
                    raise ValueError("Pool window lies outside the chosen elutions or exceeds its volume limit")
                candidate = _recipe(config, v, washes, elutions, style)
                sim_key = tuple(v[k] for k in nonpool)
                if sim_key in simulations:
                    eval_id, result, prior_error = simulations[sim_key]
                    simulations.move_to_end(sim_key)
                    reused_count += 1
                    reused = True
                    if prior_error:
                        raise RuntimeError("Cached model failure: " + prior_error)
                else:
                    if solve_count >= cap:
                        raise StopIteration("Evaluation cap")
                    solve_count += 1
                    eval_id = solve_count
                    new = True
                    deadline = (time.monotonic()+60*float(options.get("max_trial_minutes", 0))
                                if float(options.get("max_trial_minutes", 0)) > 0 else None)
                    try:
                        result = (simulate(candidate, solver_deadline=deadline) if simulator is simulate
                                  else simulator(candidate))
                    except Exception as exc:
                        simulations[sim_key] = (eval_id, None, f"{type(exc).__name__}: {exc}")
                        raise
                    simulations[sim_key] = (eval_id, result, None)
                    if len(simulations) > 10:
                        simulations.popitem(last=False)
                entry["metrics"] = _metrics(result, candidate, v, options)
                consider(entry, result)
            except StopIteration:
                raise
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            add(entry, eval_id, reused)
            if new and entry["metrics"] is not None:
                pool_best = _optimise_pool_window(
                    result, candidate, v, options,
                    (float(lower[-2]), float(upper[-2])),
                    (float(lower[-1]), float(upper[-1])), metric_key)
                if pool_best:
                    vv, mm = pool_best
                    kk = tuple(vv[k] for k in variables)
                    if kk not in cache:
                        screened = {"values": vv, "metrics": mm, "error": None,
                                    "result": None, "phase": "pool_refine"}
                        consider(screened, result)
                        add(screened, eval_id, True)
            return entry

        def score(u: np.ndarray) -> float:
            m = evaluate(u, "local")["metrics"]
            if m is None:
                return 1e6
            penalty = sum((max(0, -float(v))/max(1, abs(float(v))))**2 for v in m["margins"].values())
            return -float(m[metric_key])+1e4*penalty

        margin_names = ["minimum_product_purity_percent", "minimum_product_recovery_percent",
                        "minimum_product_mass_g", "maximum_pool_volume_mL", "pool_order_CV",
                        "pool_within_recipe_CV"]
        margin_names += [f"maximum_{name}_percent" for name in options.get("max_impurity_percent", {})]
        if options.get("max_run_time_min") is not None:
            margin_names.append("max_run_time_min")
        if options.get("max_buffer_volume_L") is not None:
            margin_names.append("max_buffer_volume_L")
        if options.get("max_pressure_bar") is not None:
            margin_names.append("maximum_pressure_bar")

        def constraint(u: np.ndarray, name: str) -> float:
            v = lower+np.clip(u, 0, 1)*(upper-lower)
            if name == "pool_order_CV":
                return float(v[-1]-v[-2]-1e-4)
            if name == "pool_within_recipe_CV":
                return float(sum(v[ix[f"el{i}_CV"]] for i in range(1, elutions+1))-v[-1])
            if name == "maximum_pool_volume_mL":
                return float(options["max_pool_volume_mL"]-(v[-1]-v[-2])*bed_mL)
            if name == "maximum_pressure_bar":
                return float(options["max_pressure_bar"]-float(options["reference_pressure_bar"])*
                             v[ix["flow_mL_min"]]/float(options["reference_flow_mL_min"]))
            m = evaluate(u, "local")["metrics"]
            return float(m["margins"][name]) if m is not None else -1e3

        baseline = {"loading_g_L": float(config["feed"]["load_density_mg_mL_resin"]),
                    "flow_mL_min": float(config["column"]["flow_mL_min"]),
                    "pool_start_CV": float(lower[-2]), "pool_end_CV": float(lower[-1])}
        for i in range(1, washes+1):
            stage = config["process"]["plw_steps"][i-1]
            baseline.update({f"plw{i}_CV": float(stage["CV"]),
                             f"plw{i}_B": float(stage.get("start_percent_B") or 0)})
        for i in range(1, elutions+1):
            stage = config["process"]["elution_steps"][i-1]
            baseline[f"el{i}_CV"] = float(stage["CV"])
            if f"el{i}_B" in variables:
                baseline[f"el{i}_B"] = float(stage.get("start_percent_B") or 0)
            else:
                baseline[f"el{i}_start_B"] = float(stage.get("start_percent_B") or 0)
                baseline[f"el{i}_end_B"] = float(stage.get("end_percent_B") or 0)
        baseline_u = np.clip(np.array([baseline[k] for k in variables]), lower, upper)
        starting = evaluate(repair((baseline_u-lower)/(upper-lower)), "baseline")
        global_count = min(max(8, cap//3), 128, cap-1)
        for u in qmc.LatinHypercube(d=len(variables), seed=73).random(global_count):
            try:
                evaluate(repair(u), "global")
            except StopIteration:
                break
        seeds = sorted((e for e in cache.values() if e["metrics"] is not None),
                       key=lambda e: e["metrics"][metric_key]
                       - 1e4*sum(max(0, -float(v)) for v in e["metrics"]["margins"].values()),
                       reverse=True)[:3]
        local_status = []
        for seed in seeds:
            if solve_count >= cap:
                break
            u = np.array([(seed["values"][k]-lower[i])/(upper[i]-lower[i])
                          for i, k in enumerate(variables)])
            try:
                fit = minimize(score, u, method="SLSQP", bounds=[(0, 1)]*len(variables),
                               constraints=[{"type": "ineq", "fun": constraint, "args": (name,)}
                                            for name in margin_names],
                               options={"maxiter": max(1, min(20, (cap-solve_count)//max(1, len(variables)-2))),
                                        "ftol": 1e-4, "eps": 1e-4})
                local_status.append({"success": bool(fit.success), "message": str(fit.message),
                                     "iterations": int(fit.nit)})
            except StopIteration:
                local_status.append({"success": False, "message": "Evaluation cap reached"})
                break

    report = {"run_name": run_name, "model": config["model"], "objective": options["objective"],
              "objective_description": OBJECTIVES[options["objective"]][1],
              "method_structure": {"plw_count": washes, "elution_count": elutions,
                                   "elution_style": style},
              "fixed_model_fingerprint": batch_parameter_fingerprint(config),
              "starting_recipe_fingerprint": configuration_fingerprint(config),
              "search_bounds": {k: [float(lower[i]), float(upper[i])] for i, k in enumerate(variables)},
              "constraints": {k: options.get(k) for k in (
                  "min_purity_percent", "min_recovery_percent", "min_product_mass_g",
                  "max_pool_volume_mL", "max_run_time_min", "max_buffer_volume_L",
                  "max_impurity_percent", "max_pressure_bar", "reference_pressure_bar",
                  "reference_flow_mL_min")},
              "starting_candidate": {k: starting.get(k) for k in ("values", "metrics", "error")},
              "evaluations": solve_count, "evaluation_cap": cap, "evaluation_cap_reached": solve_count >= cap,
              "candidates": len(cache), "simulation_cache_hits": reused_count,
              "local_search": local_status, "best_feasible": best is not None,
              "history_csv": str(history), "folder": str(folder),
              "search_status": ("EVALUATION_CAP_REACHED" if solve_count >= cap else
                                "LOCAL_SEED_CONVERGED" if any(x["success"] for x in local_status) else
                                "EXPLORATORY_SEARCH"),
              "guarantee": "Best observed feasible candidate; global optimality is not established.",
              "pressure_basis": ("Linear flow-scaled estimate from measured reference pressure; validate with column data."
                                 if options.get("max_pressure_bar") is not None else "Pressure was not constrained."),
              "cycle_basis": "Modelled load, post-load wash and elution only; cleaning, regeneration and setup are excluded.",
              "peak_resolution_basis": "Minimum product-to-impurity baseline resolution, 1.18 ΔCV/(FWHM product + FWHM impurity), from column outlet profiles. Unresolved or noneluted peaks score zero.",
              "history_rows": len(cache), "robustness_model_evaluations": 0,
              "total_model_evaluations": solve_count,
              "search_quality_note": ("The selected cap is small for this method dimension; results may be exploratory."
                                      if cap < 3*len(variables) else "")}
    if best is not None:
        v, m, result = best["values"], best["metrics"], best["result"]
        report.update(conditions=v, pool=m, objective_value=m[metric_key],
                      active_constraints=[k for k, margin in m["margins"].items()
                                          if margin <= max(1e-6, .01*max(1, abs(float(margin))))])
        candidate = _recipe(config, v, washes, elutions, style)
        (folder/"best_recipe.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        trace = result["trace"]
        names = [str(x["name"]) for x in active_components(config)]
        with (folder/"predicted_chromatogram.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["CV", "UV_mAU", "flow_mL_min", *[f"{n}_column_outlet_g_L" for n in names]])
            for i, cv in enumerate(trace["CV"]):
                writer.writerow([cv, trace["uv_mAU"][i], trace["flow_mL_min"][i],
                                 *trace["column_component_g_L"][i]])
        svg = _svg_plot(trace["CV"], [("Total UV", trace["uv_mAU"])] +
                        [(name, trace["component_uv_mAU"][:, i]) for i, name in enumerate(names)],
                        total_cv=float(trace["CV"][-1]), percent_B=trace["programmed_percent_B"],
                        flow_mL_min=trace["flow_mL_min"])
        (folder/"predicted_chromatogram.svg").write_text(svg, encoding="utf-8")
        with (folder/"pool_composition.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["species", "mass_g", "mass_percent"])
            for i, name in enumerate(names):
                writer.writerow([name, m["masses_g"][i], m["fractions_percent"][name]])
        # Fixed-recipe purity/recovery trade-off: vary only collection cuts on
        # this already simulated chromatogram (Leweke & von Lieres, Sec. 6.2.2).
        purity_levels = sorted({float(options["min_purity_percent"]), *(
            level for level in (90, 95, 97, 98, 99, 99.5, 99.9)
            if level >= float(options["min_purity_percent"]))})
        tradeoff = []
        for minimum in purity_levels:
            tradeoff_options = dict(options, min_purity_percent=minimum)
            optimum = _optimise_pool_window(
                result, candidate, v, tradeoff_options,
                (float(lower[-2]), float(upper[-2])),
                (float(lower[-1]), float(upper[-1])), "yield_percent")
            row = {"minimum_purity_percent": minimum, "feasible": optimum is not None,
                   "achieved_purity_percent": None, "product_recovery_percent": None,
                   "pool_start_CV": None, "pool_end_CV": None}
            if optimum is not None:
                cuts, tradeoff_metrics = optimum
                row.update(achieved_purity_percent=tradeoff_metrics["purity_percent"],
                           product_recovery_percent=tradeoff_metrics["yield_percent"],
                           pool_start_CV=cuts["pool_start_CV"],
                           pool_end_CV=cuts["pool_end_CV"])
            tradeoff.append(row)
        report["fixed_recipe_pool_tradeoff"] = tradeoff
        report["pool_tradeoff_basis"] = (
            "Best observed recovery at each minimum purity, varying only pool cuts on "
            "the best recipe's simulated outlet profile; all other selected constraints apply. "
            "This is not a global method Pareto frontier.")
        with (folder/"fixed_recipe_pool_tradeoff.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(tradeoff[0]))
            writer.writeheader()
            writer.writerows(tradeoff)
        robustness = []
        robustness_calls = 0
        for variation, scale in (("pH", .1), ("loading", .05), ("flow", .05),
                                 ("salt", .05), ("pool_start", .1), ("pool_end", .1)):
            for direction in (-1, 1):
                varied = dict(v)
                varied_result = result
                error = ""
                vm = None
                simulated = variation not in ("pool_start", "pool_end")
                failure_stage = "setup"
                try:
                    if variation == "loading":
                        varied["loading_g_L"] *= 1 + direction*scale
                    elif variation == "flow":
                        varied["flow_mL_min"] *= 1 + direction*scale
                    elif variation.startswith("pool_"):
                        varied[variation + "_CV"] += direction*scale
                        if varied["pool_start_CV"] < 0:
                            failure_stage = "pool_evaluation"
                            raise ValueError("Pool start cannot precede the first elution")
                    varied_config = _recipe(config, varied, washes, elutions, style)
                    if variation == "pH":
                        process = varied_config["process"]
                        for buffer in ("buffer_A", "buffer_B"):
                            process[buffer]["pH"] += direction*scale
                        for field in ("load_pH", "equilibration_pH"):
                            if process.get(field) is not None:
                                process[field] += direction*scale
                    elif variation == "salt":
                        process = varied_config["process"]
                        for buffer in ("buffer_A", "buffer_B"):
                            b = process[buffer]
                            for field in ("conductivity_mS_cm", "salt_concentration_M"):
                                if b.get(field) is not None:
                                    b[field] *= 1 + direction*scale
                            for ion in b.get("ions", []):
                                if ion.get("concentration_M") is not None:
                                    ion["concentration_M"] *= 1 + direction*scale
                        for field in ("load_conductivity_mS_cm", "load_salt_concentration_M",
                                      "equilibration_conductivity_mS_cm", "equilibration_salt_concentration_M"):
                            if process.get(field) is not None:
                                process[field] *= 1 + direction*scale
                    if simulated:
                        robustness_calls += 1
                        deadline = (time.monotonic()+60*float(options.get("max_trial_minutes", 0))
                                    if float(options.get("max_trial_minutes", 0)) > 0 else None)
                        failure_stage = "simulation"
                        varied_result = (simulate(varied_config, solver_deadline=deadline)
                                         if simulator is simulate else simulator(varied_config))
                    failure_stage = "metric_evaluation" if simulated else "pool_evaluation"
                    vm = _metrics(varied_result, varied_config, varied, options)
                    failure_stage = ""
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                observation = {"variation": variation, "direction": direction, "magnitude": scale,
                               "objective_value": vm[metric_key] if vm else None,
                               "metrics": vm, "error": error, "failure_stage": failure_stage}
                robustness.append(observation)
                with history.open("a", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=fields).writerow({
                        "candidate": len(cache)+len(robustness),
                        "model_evaluation": solve_count+robustness_calls if simulated else "",
                        "simulation_reused": not simulated, "phase": "robustness", **varied,
                        "objective_value": vm[metric_key] if vm else "",
                        **({k: vm.get(k, "") for k in fields if k in vm} if vm else {}),
                        "feasible": vm["feasible"] if vm else False,
                        "error": error, "margins_json": json.dumps(vm["margins"] if vm else {})})
        report["robustness"] = robustness
        report["robustness_model_evaluations"] = robustness_calls
        report["total_model_evaluations"] = solve_count+robustness_calls
        report["history_rows"] = len(cache)+len(robustness)
        observed = [row["metrics"] for row in robustness if row["metrics"] is not None]
        report["robustness_summary"] = {
            "tested_variations": len(robustness),
            "simulation_failures": sum(row["failure_stage"] == "simulation" for row in robustness),
            "setup_failures": sum(row["failure_stage"] == "setup" for row in robustness),
            "metric_failures": sum(row["failure_stage"] == "metric_evaluation" for row in robustness),
            "invalid_pool_perturbations": sum(row["failure_stage"] == "pool_evaluation" for row in robustness),
            "constraint_failures": sum(not row["feasible"] for row in observed),
            "total_failed_perturbations": sum(row["metrics"] is None or not row["metrics"]["feasible"]
                                              for row in robustness),
            "lowest_purity_percent": min((row["purity_percent"] for row in observed), default=None),
            "lowest_recovery_percent": min((row["yield_percent"] for row in observed), default=None),
            "basis": ("Local one-factor perturbations only; these counts are not a failure "
                      "probability or a validated confidence interval.")}
    report_html = folder/"method_optimisation_report.html"
    report["report_html"] = str(report_html)
    (folder/"method_optimisation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = (f"<h1>Method optimisation — {html.escape(run_name)}</h1>"
               f"<p>{html.escape(report['objective_description'])}; {solve_count} model evaluations. "
               f"{html.escape(report['guarantee'])}</p>"
               f"<p>Structure: {washes} washes, {elutions} elutions, {html.escape(style)}. "
               f"Status: {html.escape(report['search_status'])}. {html.escape(report['search_quality_note'])} "
               f"{html.escape(report['cycle_basis'])} "
               f"{html.escape(report['pressure_basis'])}</p>"
               f"<p>{html.escape(report['peak_resolution_basis'])}</p>"
               f"<p>Local SLSQP status: {html.escape(json.dumps(local_status))}</p>")
    if best is not None:
        metric_names = ["yield_percent", "purity_percent", "product_mass_g",
                        "productivity_g_L_resin_h", "pool_concentration_g_L",
                        "buffer_efficiency_g_L", "buffer_volume_L", "minimum_peak_resolution",
                        "run_time_min", "pool_volume_mL"]
        if m["estimated_pressure_bar"] is not None:
            metric_names.append("estimated_pressure_bar")
        summary += ("<h2>Best observed feasible method</h2><table border='1'><tr><th>Metric</th><th>Value</th></tr>" +
                    "".join(f"<tr><td>{html.escape(k)}</td><td>{m[k]:.6g}</td></tr>" for k in metric_names) + "</table>" +
                    "<h3>Operating conditions</h3><pre>"+html.escape(json.dumps(v, indent=2))+"</pre>" +
                    "<h3>Pool composition</h3><pre>"+html.escape(json.dumps(m["fractions_percent"], indent=2))+"</pre>" +
                    "<h3>Peak resolutions</h3><pre>"+html.escape(json.dumps(m["peak_resolutions"], indent=2))+"</pre>" +
                    "<h3>Constraint margins and active limits</h3><pre>"+
                    html.escape(json.dumps({"active": report["active_constraints"],
                                            "margins": m["margins"]}, indent=2))+"</pre>" +
                    "<h3>Fixed-recipe pool purity/recovery trade-off</h3><p>"+
                    html.escape(report["pool_tradeoff_basis"])+"</p><p>"+
                    "<a href='fixed_recipe_pool_tradeoff.csv'>Download trade-off CSV</a></p><pre>"+
                    html.escape(json.dumps(tradeoff, indent=2))+"</pre>"+
                    "<h3>Local robustness</h3><pre>"+
                    html.escape(json.dumps(report["robustness_summary"], indent=2))+"</pre><pre>"+
                    html.escape(json.dumps(robustness, indent=2))+"</pre>" +
                    "<img style='max-width:100%' src='predicted_chromatogram.svg'>")
    else:
        summary += "<p>No feasible method found within the selected bounds and evaluation cap.</p>"
    summary += "<p><a href='evaluation_history.csv'>Evaluation history</a> | <a href='method_optimisation_report.json'>Full report</a></p>"
    report_html.write_text("<!doctype html><html><meta charset='utf-8'><body>"+summary+"</body></html>", encoding="utf-8")
    return report
