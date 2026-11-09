"""Bounded operating-condition search around a fixed mechanistic model.

This module only calls the production simulator. It never fits or changes
mechanistic parameters, the selected recipe, or LSQ inputs.
"""
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

from tdm_minimal_io import (active_components, batch_parameter_fingerprint,
                            configuration_fingerprint, normalize_config, validate_config)
from tdm_minimal_model import simulate, _svg_plot


VARIABLES = ("loading_g_L", "flow_mL_min", "wash_CV", "gradient_start_B",
             "gradient_end_B", "gradient_CV", "pool_start_CV", "pool_end_CV")


def _gradient_index(config: dict[str, Any]) -> int:
    process = config["process"]
    return next(i for i, stage in enumerate(process["elution_steps"][:int(process["elution_count"])])
                if stage["mode"] == "LINEAR" and stage["chemistry_control"] == "BUFFER_B_PERCENT")


def _recipe(config: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    p = result["process"]
    load = values["loading_g_L"]
    flow = values["flow_mL_min"]
    result["feed"]["load_density_mg_mL_resin"] = load
    p["load_amount_basis"] = "CAPACITY"
    p["load_CV"] = load / float(result["feed"]["total_concentration_mg_mL"])
    p["load_volume_mL"] = p["load_CV"] * float(result["column"]["volume_mL"])
    result["column"]["flow_mL_min"] = flow
    for key, count in (("plw_steps", "plw_count"), ("elution_steps", "elution_count")):
        for stage in p[key][:int(p[count])]:
            stage["flow_mL_min"] = flow
    if int(p["plw_count"]) > 0:
        p["plw_steps"][0]["CV"] = values["wash_CV"]
    gradient = p["elution_steps"][_gradient_index(result)]
    gradient["start_percent_B"] = values["gradient_start_B"]
    gradient["end_percent_B"] = values["gradient_end_B"]
    gradient["CV"] = values["gradient_CV"]
    return result


def _pool_masses(trace: dict[str, Any], start: float, end: float, volume_mL: float) -> np.ndarray:
    cv = np.asarray(trace["CV"], dtype=float)
    concentrations = np.maximum(np.asarray(trace["column_component_g_L"], dtype=float), 0.0)
    if (not np.isfinite(cv).all() or not np.isfinite(concentrations).all()
            or not np.all(np.diff(cv) > 0)):
        raise FloatingPointError("Nonfinite or unordered column outlet profile")
    if not 0 <= start < end <= cv[-1]:
        raise ValueError("Pool boundaries must lie inside the simulated chromatogram.")
    inner = cv[(cv > start) & (cv < end)]
    points = np.r_[start, inner, end]
    sampled = np.column_stack([np.interp(points, cv, concentrations[:, i])
                               for i in range(concentrations.shape[1])])
    # c [g/L] * Q [mL/min] * dt [min] = c * dV [mL]/1000.
    # CV is cumulative volume / V_col, so this is equivalent to c * V_col * dCV.
    return np.trapezoid(sampled, points, axis=0) * volume_mL / 1000.0


def _elution_start(result: dict[str, Any], config: dict[str, Any]) -> float:
    program = result["simulation"]["program"]
    stage_index = 1 + int(config["process"]["plw_count"]) + _gradient_index(config)
    stage = program.stages[stage_index]
    if stage.kind != "ELUTION" or stage.mode != "LINEAR":
        raise ValueError("Selected %B gradient did not match the simulated stage order")
    return float(program.stage_start_CV[stage_index])


def _available_pool_span(config: dict[str, Any]) -> float:
    process = config["process"]
    steps = process["elution_steps"][:int(process["elution_count"])]
    first = _gradient_index(config)
    return sum(float(stage["CV"]) for stage in steps[first:])


def _metrics(result: dict[str, Any], config: dict[str, Any], values: dict[str, float],
             limits: dict[str, Any]) -> dict[str, Any]:
    start = _elution_start(result, config) + values["pool_start_CV"]
    end = _elution_start(result, config) + values["pool_end_CV"]
    masses = _pool_masses(result["trace"], start, end, float(config["column"]["volume_mL"]))
    total = float(masses.sum())
    loaded = (float(config["feed"]["load_density_mg_mL_resin"])
              * float(config["column"]["volume_mL"]) / 1000.0
              * float(active_components(config)[0]["mass_percent"]) / 100.0)
    product = float(masses[0])
    purity = 100.0 * product / total if total > 0 else 0.0
    yield_pct = 100.0 * product / loaded if loaded > 0 else 0.0
    species = active_components(config)
    fractions = {str(species[i]["name"]): 100.0 * float(m) / total if total > 0 else 0.0
                 for i, m in enumerate(masses)}
    pressure = None
    if limits.get("max_pressure_bar") is not None:
        pressure = (float(limits["reference_pressure_bar"]) * values["flow_mL_min"]
                    / float(limits["reference_flow_mL_min"]))
    margins = {"minimum_product_purity_percent": purity - float(limits["min_purity_percent"]),
               "maximum_pool_volume_mL": float(limits["max_pool_volume_mL"])
                 - (end - start) * float(config["column"]["volume_mL"]),
               "pool_order_CV": values["pool_end_CV"] - values["pool_start_CV"] - 1e-4,
               "pool_within_recipe_CV": _available_pool_span(config) - values["pool_end_CV"]}
    for i, item in enumerate(species[1:], start=1):
        name = str(item["name"])
        cap = limits.get("max_impurity_percent", {}).get(name)
        if cap is not None:
            margins[f"maximum_{name}_percent"] = float(cap) - fractions[name]
    if pressure is not None:
        margins["maximum_pressure_bar"] = float(limits["max_pressure_bar"]) - pressure
    program = result["simulation"]["program"]
    run_time_min = float(program.total_time_s) / 60.0
    pool_start_min = float(program.cv_to_time(start)) / 60.0
    pool_end_min = float(program.cv_to_time(end)) / 60.0
    bed_volume_L = float(config["column"]["volume_mL"]) / 1000.0
    return {"yield_percent": yield_pct, "purity_percent": purity, "masses_g": masses.tolist(),
            "fractions_percent": fractions, "loaded_product_g": loaded,
            "pool_start_CV": start, "pool_end_CV": end,
            "pool_start_min": pool_start_min, "pool_end_min": pool_end_min,
            "run_time_min": run_time_min,
            "productivity_g_L_resin_h": (product / (bed_volume_L * run_time_min / 60.0)
                                         if run_time_min > 0 else 0.0),
            "pool_volume_mL": (end-start)*float(config["column"]["volume_mL"]),
            "estimated_pressure_bar": pressure, "margins": margins,
            "feasible": bool(all(value >= -1e-6 for value in margins.values()))}


def _screen_pool_windows(result: dict[str, Any], config: dict[str, Any],
                         values: dict[str, float], limits: dict[str, Any],
                         lower: np.ndarray, upper: np.ndarray,
                         grid_size: int = 17) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Find a feasible pool for one column solution without solving it again."""
    available = min(_available_pool_span(config),
                    float(result["trace"]["CV"][-1]) - _elution_start(result, config))
    max_span = float(limits["max_pool_volume_mL"]) / float(config["column"]["volume_mL"])
    start_high = min(float(upper[6]), available - 1.01e-4)
    end_high = min(float(upper[7]), available)
    if start_high < lower[6] or end_high < lower[7]:
        return None
    starts = np.linspace(float(lower[6]), start_high, grid_size)
    ends = np.linspace(float(lower[7]), end_high, grid_size)
    # Check the current window and exact volume/recipe boundaries as well as
    # the grid; these often determine the best feasible collection window.
    pairs = [(values["pool_start_CV"], values["pool_end_CV"])]
    for start in starts:
        pairs.extend((float(start), float(end)) for end in ends)
        pairs.append((float(start), min(end_high, float(start) + max_span)))
    best_pair: tuple[dict[str, float], dict[str, Any]] | None = None
    for start, end in pairs:
        if (not float(lower[6]) <= start <= float(upper[6])
                or not float(lower[7]) <= end <= end_high
                or end - start <= 1.01e-4
                or end - start > max_span + 1e-9):
            continue
        trial_values = dict(values, pool_start_CV=start, pool_end_CV=end)
        try:
            trial = _metrics(result, config, trial_values, limits)
        except (ValueError, FloatingPointError):
            continue
        if trial["feasible"] and (best_pair is None or
                                  trial["yield_percent"] > best_pair[1]["yield_percent"]):
            best_pair = (trial_values, trial)
    return best_pair


def _check_options(config: dict[str, Any], options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    validation_errors = validate_config(config, strict=True)
    if validation_errors:
        raise ValueError("Complete and validate the selected run before method optimisation: " + "; ".join(validation_errors))
    species_names = [str(row["name"]).strip() for row in active_components(config)]
    if any(not name for name in species_names) or len(set(species_names)) != len(species_names):
        raise ValueError("Method optimisation requires a distinct nonblank name for each product and impurity.")
    p = config["process"]
    gradients = [s for s in p["elution_steps"][:int(p["elution_count"])]
                 if s["mode"] == "LINEAR" and s["chemistry_control"] == "BUFFER_B_PERCENT"]
    if not gradients:
        raise ValueError("Method optimisation needs a linear %B elution step.")
    if int(p["plw_count"]) < 1:
        raise ValueError("Method optimisation needs one post-load wash to vary its duration.")
    bounds = options.get("bounds", {})
    try:
        lower = np.array([float(bounds[name][0]) for name in VARIABLES])
        upper = np.array([float(bounds[name][1]) for name in VARIABLES])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Enter lower and upper bounds for all eight method variables.") from exc
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()
            and np.all(lower < upper) and np.all(lower[:3] > 0)
            and np.all(lower[3:5] >= 0) and np.all(upper[3:5] <= 100)
            and lower[5] > 0 and lower[6] >= 0 and lower[7] > lower[6]):
        raise ValueError("Method bounds must be finite and ordered; load, flow, wash and gradient length must be positive, %B 0–100, and pool end after pool start.")
    try:
        min_purity = float(options["min_purity_percent"])
        max_pool_volume = float(options["max_pool_volume_mL"])
        raw_cap = float(options.get("max_evaluations", 30))
        timeout = float(options.get("max_trial_minutes", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Enter numeric minimum purity, maximum pool volume, evaluation cap and trial timeout.") from exc
    if not math.isfinite(min_purity) or min_purity <= 0:
        raise ValueError("Minimum purity must be positive and finite.")
    if not math.isfinite(max_pool_volume) or max_pool_volume <= 0:
        raise ValueError("Maximum pool volume must be positive and finite.")
    if min_purity > 100:
        raise ValueError("Minimum purity must be at most 100%.")
    max_pool_cv = max_pool_volume / float(config["column"]["volume_mL"])
    first = _gradient_index(config)
    following_cv = sum(float(s["CV"]) for s in
                       p["elution_steps"][first + 1:int(p["elution_count"])])
    available_max = upper[5] + following_cv
    start_low = max(lower[6], lower[7] - max_pool_cv)
    start_high = min(upper[6], available_max - 1.01e-4, upper[7] - 1.01e-4)
    if (max_pool_cv <= 1.01e-4 or lower[7] > available_max + 1e-9
            or start_low > start_high):
        raise ValueError("Pool bounds and volume limit leave no collection window within the simulated elution program.")
    impurity_names = {str(row["name"]) for row in active_components(config)[1:]}
    for name, cap in options.get("max_impurity_percent", {}).items():
        if name not in impurity_names:
            raise ValueError(f"Impurity cap refers to unknown species: {name}")
        if not math.isfinite(float(cap)) or not 0 <= float(cap) <= 100:
            raise ValueError("Impurity caps must be between 0 and 100%.")
    pressure_keys = ("max_pressure_bar", "reference_pressure_bar", "reference_flow_mL_min")
    if any(options.get(key) is not None for key in pressure_keys):
        for key in pressure_keys:
            if options.get(key) is None or not math.isfinite(float(options[key])) or float(options[key]) <= 0:
                raise ValueError("A pressure cap requires positive measured reference pressure and reference flow. The pressure estimate scales linearly with flow for this column and liquid.")
        lowest_pressure = (float(options["reference_pressure_bar"]) * lower[1]
                           / float(options["reference_flow_mL_min"]))
        if lowest_pressure > float(options["max_pressure_bar"]) + 1e-9:
            raise ValueError("Every allowed flow violates the selected pressure cap under the flow-scaled estimate.")
    if not math.isfinite(raw_cap) or raw_cap != int(raw_cap) or not 2 <= raw_cap <= 1000:
        raise ValueError("Method evaluation cap must be 2–1000.")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Method trial timeout must be nonnegative; 0 disables it.")
    return lower, upper


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")[:48] or "Run"


class _Budget(RuntimeError):
    pass


def optimise_method(config: dict[str, Any], options: dict[str, Any], *,
                    run_name: str = "Run", output_root: Path | str = "Method_Optimisation",
                    simulator: Callable[[dict[str, Any]], dict[str, Any]] = simulate) -> dict[str, Any]:
    """Return the best *observed feasible* candidate within a hard simulation cap."""
    config = normalize_config(config)
    lower, upper = _check_options(config, options)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"{_safe_name(run_name)}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir()
    history_path = folder / "evaluation_history.csv"
    fields = ["candidate", "model_evaluation", "simulation_reused", "phase",
              "variation", "direction", "magnitude",
              *VARIABLES, "yield_percent", "purity_percent",
              "feasible", "error", "margins_json"]
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    simulation_cache: OrderedDict[tuple[float, ...], tuple[int, dict[str, Any] | None, str | None]] = OrderedDict()
    simulation_cache_limit = 10
    simulation_cache_hits = 0
    best: dict[str, Any] | None = None
    solve_count = 0
    cap = int(options.get("max_evaluations", 30))
    first_gradient = _gradient_index(config)
    following_cv = sum(float(stage["CV"]) for stage in
                       config["process"]["elution_steps"][first_gradient + 1:
                           int(config["process"]["elution_count"])])
    max_pool_cv = float(options["max_pool_volume_mL"]) / float(config["column"]["volume_mL"])

    def repair_pool_seed(unit: np.ndarray) -> np.ndarray:
        """Sample valid pool geometry without changing the local objective."""
        physical = lower + np.clip(np.asarray(unit, dtype=float), 0, 1) * (upper - lower)
        if options.get("max_pressure_bar") is not None:
            max_flow = (float(options["max_pressure_bar"]) * float(options["reference_flow_mL_min"])
                        / float(options["reference_pressure_bar"]))
            physical[1] = min(physical[1], max_flow)
        available = physical[5] + following_cv
        epsilon = 1.01e-4
        start_low = max(lower[6], lower[7] - max_pool_cv)
        start_high = min(upper[6], available - epsilon, upper[7] - epsilon)
        if start_low > start_high:
            return np.asarray(unit, dtype=float)
        physical[6] = np.clip(physical[6], start_low, start_high)
        end_low = max(lower[7], physical[6] + epsilon)
        end_high = min(upper[7], available, physical[6] + max_pool_cv)
        if end_low > end_high:
            return np.asarray(unit, dtype=float)
        physical[7] = np.clip(physical[7], end_low, end_high)
        return (physical - lower) / (upper - lower)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def evaluate(unit: np.ndarray, phase: str) -> dict[str, Any]:
            nonlocal best, solve_count, simulation_cache_hits
            unit = np.clip(np.asarray(unit, dtype=float), 0, 1)
            values = dict(zip(VARIABLES, (lower + unit * (upper - lower)).tolist()))
            key = tuple(values[name] for name in VARIABLES)
            if key in cache:
                return cache[key]
            entry: dict[str, Any] = {"values": values, "metrics": None, "error": None,
                                     "result": None, "phase": phase}
            reused = False
            new_simulation = False
            evaluation_id: int | None = None
            try:
                if values["pool_end_CV"] <= values["pool_start_CV"]:
                    raise ValueError("Pool end precedes pool start")
                candidate = _recipe(config, values)
                if values["pool_end_CV"] > _available_pool_span(candidate) + 1e-9:
                    raise ValueError("Pool end lies beyond the selected recipe; skipped without a simulation")
                method_key = tuple(values[name] for name in VARIABLES[:6])
                if method_key in simulation_cache:
                    evaluation_id, result, prior_error = simulation_cache[method_key]
                    simulation_cache.move_to_end(method_key)
                    simulation_cache_hits += 1
                    reused = True
                    if prior_error is not None:
                        raise RuntimeError("Cached model failure: " + prior_error)
                else:
                    if solve_count >= cap:
                        raise _Budget()
                    solve_count += 1
                    evaluation_id = solve_count
                    new_simulation = True
                    deadline = (time.monotonic() + 60 * float(options.get("max_trial_minutes", 0))
                                if float(options.get("max_trial_minutes", 0)) > 0 else None)
                    try:
                        result = (simulate(candidate, solver_deadline=deadline)
                                  if simulator is simulate else simulator(candidate))
                    except Exception as exc:
                        simulation_cache[method_key] = (evaluation_id, None, f"{type(exc).__name__}: {exc}")
                        if len(simulation_cache) > simulation_cache_limit:
                            simulation_cache.popitem(last=False)
                        raise
                    simulation_cache[method_key] = (evaluation_id, result, None)
                    if len(simulation_cache) > simulation_cache_limit:
                        simulation_cache.popitem(last=False)
                metrics = _metrics(result, candidate, values, options)
                entry["metrics"] = metrics
                if metrics["feasible"] and (best is None or metrics["yield_percent"] > best["metrics"]["yield_percent"]):
                    if best is not None:
                        best["result"] = None
                    entry["result"] = result
                    best = entry
            except _Budget:
                raise
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            cache[key] = entry
            metrics = entry["metrics"] or {}
            writer.writerow({"candidate": len(cache), "model_evaluation": evaluation_id or "",
                             "simulation_reused": reused,
                             "phase": phase, **values,
                             "yield_percent": metrics.get("yield_percent", ""),
                             "purity_percent": metrics.get("purity_percent", ""),
                             "feasible": metrics.get("feasible", False),
                             "error": entry["error"] or "",
                             "margins_json": json.dumps(metrics.get("margins", {}))})
            handle.flush()
            if new_simulation and entry["metrics"] is not None:
                refined = _screen_pool_windows(result, candidate, values, options, lower, upper)
                if refined is not None:
                    refined_values, refined_metrics = refined
                    refined_key = tuple(refined_values[name] for name in VARIABLES)
                    if refined_key not in cache:
                        refined_entry = {"values": refined_values, "metrics": refined_metrics,
                                         "error": None, "result": None, "phase": "pool_screen"}
                        cache[refined_key] = refined_entry
                        if best is None or refined_metrics["yield_percent"] > best["metrics"]["yield_percent"]:
                            if best is not None:
                                best["result"] = None
                            refined_entry["result"] = result
                            best = refined_entry
                        writer.writerow({"candidate": len(cache), "model_evaluation": evaluation_id or "",
                                         "simulation_reused": True, "phase": "pool_screen",
                                         **refined_values,
                                         "yield_percent": refined_metrics["yield_percent"],
                                         "purity_percent": refined_metrics["purity_percent"],
                                         "feasible": True, "error": "",
                                         "margins_json": json.dumps(refined_metrics["margins"])})
                        handle.flush()
            return entry

        def score(unit: np.ndarray) -> float:
            e = evaluate(unit, "local")
            m = e["metrics"]
            if m is None:
                return 1e5
            penalty = sum(max(0.0, -float(v))**2 for v in m["margins"].values())
            return -float(m["yield_percent"]) + 1e3 * penalty

        constraint_names = ["minimum_product_purity_percent", "maximum_pool_volume_mL",
                            "pool_order_CV", "pool_within_recipe_CV"]
        constraint_names += [f"maximum_{name}_percent" for name in options.get("max_impurity_percent", {})]
        if options.get("max_pressure_bar") is not None:
            constraint_names.append("maximum_pressure_bar")

        def constraint(unit: np.ndarray, name: str) -> float:
            physical = lower + np.clip(np.asarray(unit, dtype=float), 0, 1) * (upper - lower)
            if name == "pool_order_CV":
                return float(physical[7] - physical[6] - 1e-4)
            if name == "pool_within_recipe_CV":
                return float(physical[5] + following_cv - physical[7])
            if name == "maximum_pool_volume_mL":
                return (float(options["max_pool_volume_mL"])
                        - float(physical[7] - physical[6]) * float(config["column"]["volume_mL"]))
            if name == "maximum_pressure_bar":
                return (float(options["max_pressure_bar"])
                        - float(options["reference_pressure_bar"]) * float(physical[1])
                        / float(options["reference_flow_mL_min"]))
            m = evaluate(unit, "local")["metrics"]
            if m is None:
                return -100.0
            scale = (100.0 if name.endswith("percent") else
                     float(options["max_pool_volume_mL"]) if name == "maximum_pool_volume_mL" else
                     float(options["max_pressure_bar"]) if name == "maximum_pressure_bar" else 1.0)
            return float(m["margins"][name]) / max(scale, 1e-9)

        gradient_baseline = next(s for s in config["process"]["elution_steps"][:int(config["process"]["elution_count"])]
                                 if s["mode"] == "LINEAR" and s["chemistry_control"] == "BUFFER_B_PERCENT")
        baseline = {"loading_g_L": float(config["feed"]["load_density_mg_mL_resin"]),
                    "flow_mL_min": float(config["column"]["flow_mL_min"]),
                    "wash_CV": float(config["process"]["plw_steps"][0]["CV"]),
                    "gradient_start_B": float(gradient_baseline["start_percent_B"]),
                    "gradient_end_B": float(gradient_baseline["end_percent_B"]),
                    "gradient_CV": float(gradient_baseline["CV"]),
                    "pool_start_CV": lower[6],
                    "pool_end_CV": max(lower[7], min(upper[7], float(gradient_baseline["CV"])*.9))}
        baseline_unit = np.clip((np.array([baseline[k] for k in VARIABLES]) - lower) / (upper-lower), 0, 1)
        starting_entry = evaluate(repair_pool_seed(baseline_unit), "baseline")
        global_count = min(max(8, cap // 3), 128, cap - 1)
        design = qmc.LatinHypercube(d=len(VARIABLES), seed=73).random(global_count)
        budget_hit = False
        local_status = []
        local_starts_attempted = 0
        for point in design:
            try:
                evaluate(repair_pool_seed(point), "global")
            except _Budget:
                budget_hit = True
                break
        seeds = sorted(((v["metrics"]["yield_percent"] - 1000*sum(max(0.0,-m)**2 for m in v["metrics"]["margins"].values()),
                         np.array([(value - lower[i]) / (upper[i] - lower[i])
                                   for i, value in enumerate(key)]))
                        for key, v in cache.items() if v["metrics"] is not None),
                       key=lambda pair: pair[0], reverse=True)[:3]
        for seed_index, (_, seed) in enumerate(seeds):
            if solve_count >= cap:
                budget_hit = True
                break
            local_starts_attempted += 1
            try:
                remaining = cap - solve_count
                starts_left = len(seeds) - seed_index
                # Pool-boundary finite differences reuse the same column solve.
                local_maxiter = max(1, min(100, remaining // (7 * starts_left)))
                fit = minimize(score, seed, method="SLSQP", bounds=[(0, 1)]*len(VARIABLES),
                               constraints=[{"type": "ineq", "fun": constraint, "args": (name,)}
                                            for name in constraint_names],
                               options={"maxiter": local_maxiter, "ftol": 1e-4, "eps": 1e-4})
                local_status.append({"success": bool(fit.success), "message": str(fit.message),
                                     "maxiter": local_maxiter, "iterations": int(fit.nit)})
            except _Budget:
                budget_hit = True
                local_status.append({"success": False, "message": "Evaluation cap reached during local refinement"})
                break

    cap_reached = budget_hit or solve_count >= cap
    local_converged = any(item["success"] for item in local_status)
    search_status = ("LOCAL_SEED_CONVERGED_SEARCH_CAPPED" if local_converged and cap_reached else
                     "LOCAL_SEED_CONVERGED" if local_converged else
                     "EVALUATION_CAP_REACHED" if cap_reached else
                     "LOCAL_NOT_CONVERGED" if local_status else "GLOBAL_EXPLORATION_ONLY")
    report: dict[str, Any] = {"run_name": run_name, "model": config["model"],
        "fixed_model_fingerprint": batch_parameter_fingerprint(config),
        "starting_recipe_fingerprint": configuration_fingerprint(config),
        "objective": "MAXIMISE_PRODUCT_YIELD_PERCENT_OF_LOADED_PRODUCT",
        "search_bounds": {name: [float(lower[i]), float(upper[i])]
                          for i, name in enumerate(VARIABLES)},
        "constraints": {key: options.get(key) for key in (
            "min_purity_percent", "max_pool_volume_mL", "max_impurity_percent",
            "max_pressure_bar", "reference_pressure_bar", "reference_flow_mL_min")},
        "starting_candidate": {"conditions": starting_entry["values"],
                               "metrics": starting_entry["metrics"],
                               "error": starting_entry["error"]},
        "evaluations": solve_count, "candidates": len(cache),
        "global_samples_planned": global_count,
        "global_samples_evaluated": sum(v["phase"] == "global" for v in cache.values()),
        "pool_screening_grid_size": 17,
        "pool_screening_candidates_added": sum(v["phase"] == "pool_screen" for v in cache.values()),
        "local_starts_attempted": local_starts_attempted,
        "max_trial_minutes_TDM_CPA": float(options.get("max_trial_minutes", 0)),
        "evaluation_cap": cap, "evaluation_cap_reached": cap_reached,
        "search_status": search_status, "local_search": local_status, "best_feasible": best is not None,
        "retained_full_simulations": len({id(result) for _, result, _ in simulation_cache.values()
                                           if result is not None} |
                                          ({id(best["result"])} if best is not None and best["result"] is not None else set())),
        "simulation_cache_limit": simulation_cache_limit,
        "simulation_cache_hits": simulation_cache_hits,
        "robustness_model_evaluations": 0, "total_model_evaluations": solve_count,
        "history_rows": len(cache),
        "history_csv": str(history_path), "folder": str(folder),
        "pressure_basis": ("Linear flow-scaled estimate from measured reference pressure; verify against column pressure data."
                           if options.get("max_pressure_bar") is not None else
                           "Pressure was not constrained because no measured reference pressure and flow were supplied."),
        "guarantee": ("Best feasible candidate found; global optimality is not established."
                       if best is not None else
                       "No feasible candidate found within the selected bounds and evaluation cap; global optimality is not established."),
        "search_quality_note": ("The selected search cap is small for eight variables and may allow only one local SLSQP step."
                                if cap < 50 else "")}
    if best is not None:
        values, metrics, result = best["values"], best["metrics"], best["result"]
        def active_tolerance(name: str) -> float:
            if name == "minimum_product_purity_percent":
                return max(.01, .01 * float(options["min_purity_percent"]))
            if name == "maximum_pool_volume_mL":
                return max(.001, .01 * float(options["max_pool_volume_mL"]))
            if name == "maximum_pressure_bar":
                return max(.001, .01 * float(options["max_pressure_bar"]))
            if name == "pool_order_CV":
                return .001
            if name == "pool_within_recipe_CV":
                return .001
            species_name = name.removeprefix("maximum_").removesuffix("_percent")
            return max(.01, .01 * float(options["max_impurity_percent"][species_name]))
        report.update(conditions=values, pool=metrics,
                      active_constraints=[key for key, margin in metrics["margins"].items()
                                          if margin <= active_tolerance(key)])
        (folder / "best_recipe.json").write_text(json.dumps(_recipe(config, values), indent=2), encoding="utf-8")
        trace = result["trace"]
        names = [str(item["name"]) for item in active_components(config)]
        with (folder / "predicted_chromatogram.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["CV", "UV_mAU", "flow_mL_min",
                             *[f"{name}_column_outlet_g_L" for name in names],
                             *[f"{name}_detector_g_L" for name in names]])
            for i, cv in enumerate(trace["CV"]):
                writer.writerow([cv, trace["uv_mAU"][i], trace["flow_mL_min"][i],
                                 *trace["column_component_g_L"][i],
                                 *trace["component_g_L"][i]])
        svg = _svg_plot(trace["CV"],
                        [("Total UV", trace["uv_mAU"])] +
                        [(name, trace["component_uv_mAU"][:, i]) for i, name in enumerate(names)],
                        total_cv=float(trace["CV"][-1]),
                        percent_B=trace["programmed_percent_B"],
                        flow_mL_min=trace["flow_mL_min"])
        (folder / "predicted_chromatogram.svg").write_text(svg, encoding="utf-8")
        robustness = []
        robustness_calls = 0
        for name, scale in (("pH", .1), ("loading", .05), ("flow", .05),
                            ("salt", .05), ("pool_start", .1), ("pool_end", .1)):
            for sign in (-1, 1):
                varied = dict(values)
                simulated = False
                try:
                    if name == "pH":
                        varied_config = _recipe(config, varied)
                        for buf in ("buffer_A", "buffer_B"):
                            varied_config["process"][buf]["pH"] += sign*scale
                        process = varied_config["process"]
                        for field in ("load_pH", "equilibration_pH"):
                            if process.get(field) is not None:
                                process[field] += sign*scale
                        for key, count in (("plw_steps", "plw_count"), ("elution_steps", "elution_count")):
                            for stage in process[key][:int(process[count])]:
                                for field in ("start_pH", "end_pH"):
                                    if stage.get(field) is not None:
                                        stage[field] += sign*scale
                    elif name == "loading":
                        varied["loading_g_L"] *= 1 + sign*scale
                        varied_config = _recipe(config, varied)
                    elif name == "flow":
                        varied["flow_mL_min"] *= 1 + sign*scale
                        varied_config = _recipe(config, varied)
                    elif name == "salt":
                        varied_config = _recipe(config, varied)
                        process = varied_config["process"]
                        for buf in ("buffer_A", "buffer_B"):
                            b = process[buf]
                            for field in ("conductivity_mS_cm", "salt_concentration_M"):
                                if b.get(field) is not None:
                                    b[field] *= 1 + sign*scale
                            for ion in b.get("ions", []):
                                if ion.get("concentration_M") is not None:
                                    ion["concentration_M"] *= 1 + sign*scale
                        for field in ("load_conductivity_mS_cm", "load_salt_concentration_M",
                                      "equilibration_conductivity_mS_cm", "equilibration_salt_concentration_M"):
                            if process.get(field) is not None:
                                process[field] *= 1 + sign*scale
                        for key, count in (("plw_steps", "plw_count"), ("elution_steps", "elution_count")):
                            for stage in process[key][:int(process[count])]:
                                for field in ("start_conductivity_mS_cm", "end_conductivity_mS_cm",
                                              "start_salt_concentration_M", "end_salt_concentration_M"):
                                    if stage.get(field) is not None:
                                        stage[field] *= 1 + sign*scale
                    else:
                        varied[name + "_CV"] += sign*scale
                        varied_config = None
                    if varied_config is not None:
                        robustness_calls += 1
                        simulated = True
                    varied_result = (result if varied_config is None else
                                     (simulate(varied_config, solver_deadline=(time.monotonic() + 60 * float(options.get("max_trial_minutes", 0))
                                                                         if float(options.get("max_trial_minutes", 0)) > 0 else None))
                                      if simulator is simulate else simulator(varied_config)))
                    varied_metrics = _metrics(varied_result, varied_config or _recipe(config, varied), varied, options)
                    robustness.append({"variation": name, "direction": sign, "magnitude": scale,
                                       "yield_percent": varied_metrics["yield_percent"],
                                       "purity_percent": varied_metrics["purity_percent"],
                                       "feasible": varied_metrics["feasible"],
                                       "margins": varied_metrics["margins"]})
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    robustness.append({"variation": name, "direction": sign, "magnitude": scale,
                                       "error": f"{type(exc).__name__}: {exc}"})
                observation = robustness[-1]
                with history_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=fields).writerow({
                        "candidate": len(cache) + len(robustness),
                        "model_evaluation": solve_count + robustness_calls if simulated else "",
                        "simulation_reused": not simulated,
                        "phase": "robustness", "variation": name, "direction": sign,
                        "magnitude": scale, **varied,
                        "yield_percent": observation.get("yield_percent", ""),
                        "purity_percent": observation.get("purity_percent", ""),
                        "feasible": observation.get("feasible", False),
                        "error": observation.get("error", ""),
                        "margins_json": json.dumps(observation.get("margins", {})),
                    })
        report["robustness"] = robustness
        report["robustness_model_evaluations"] = robustness_calls
        report["total_model_evaluations"] = solve_count + robustness_calls
        report["history_rows"] = len(cache) + len(robustness)
        with (folder / "pool_composition.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["species", "mass_g", "mass_percent"])
            for i, name in enumerate(names):
                writer.writerow([name, metrics["masses_g"][i], metrics["fractions_percent"][name]])
    html_path = folder / "method_optimisation_report.html"
    report["report_html"] = str(html_path)
    report_path = folder / "method_optimisation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    summary = (f"<h1>Method optimisation — {html.escape(run_name)}</h1>"
               f"<p>{html.escape(report['guarantee'])} {report['evaluations']} simulations; "
               f"cap {cap}; cap reached: {report['evaluation_cap_reached']}. "
               f"Candidates checked: {len(cache)}. Robustness simulations: "
               f"{report['robustness_model_evaluations']}; total model simulations: "
               f"{report['total_model_evaluations']}. {html.escape(report['pressure_basis'])}</p>"
               f"<p>Search status: {html.escape(search_status)}; "
               f"feasible candidate found: {best is not None}.</p>"
               f"<p>Fixed model fingerprint: {html.escape(report['fixed_model_fingerprint'])}. "
               f"{html.escape(report['search_quality_note'])}</p>"
               f"<p>Objective: maximise product yield from loaded product mass. "
               f"Global samples evaluated: {report['global_samples_evaluated']} of {global_count} planned; "
               f"local starts attempted: {local_starts_attempted}.</p>"
               f"<p>Local SLSQP status: {html.escape(json.dumps(local_status))}. "
               "A feasible candidate is reported independently of the local convergence flag.</p>")
    if best is not None:
        summary += (f"<p>Best feasible product yield: {metrics['yield_percent']:.4g}%; "
                    f"purity: {metrics['purity_percent']:.4g}%; pool "
                    f"{metrics['pool_start_CV']:.4g}–{metrics['pool_end_CV']:.4g} CV "
                    f"({metrics['pool_start_min']:.4g}–{metrics['pool_end_min']:.4g} min). "
                    f"Run time: {metrics['run_time_min']:.4g} min; product collected: "
                    f"{metrics['masses_g'][0]:.4g} g; productivity: "
                    f"{metrics['productivity_g_L_resin_h']:.4g} g/L resin/h.</p>"
                    + "<h2>Operating conditions</h2><table>"
                    + "".join(f"<tr><th>{html.escape(k)}</th><td>{v:.6g}</td></tr>" for k,v in values.items())
                    + "</table><h2>Pool composition</h2><table>"
                    + "".join(f"<tr><th>{html.escape(k)}</th><td>{v:.4g}%</td></tr>"
                              for k,v in metrics["fractions_percent"].items()) + "</table>"
                    + "<h2>Constraint margins</h2><table>"
                    + "".join(f"<tr><th>{html.escape(k)}</th><td>{v:.6g}</td></tr>"
                              for k,v in metrics["margins"].items()) + "</table>"
                    + "<p>Active constraints: " + html.escape(", ".join(report["active_constraints"]) or "none") + "</p>"
                    + '<img src="predicted_chromatogram.svg" width="900"/>')
        summary += ("<h2>Local robustness</h2><table><tr><th>Variation</th><th>Direction</th><th>Product yield [%]</th>"
                    "<th>Purity [%]</th><th>Feasible</th><th>Error</th></tr>"
                    + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                              (row["variation"], row["direction"], row.get("yield_percent", ""),
                               row.get("purity_percent", ""), row.get("feasible", ""), row.get("error", ""))) + "</tr>"
                              for row in report["robustness"]) + "</table>")
    else:
        summary += "<p>No feasible candidate was found within the simulation cap and selected bounds.</p>"
    summary += ('<p>Files: <a href="evaluation_history.csv">full evaluation history</a>, '
                '<a href="method_optimisation_report.json">report data</a>'
                + (', <a href="predicted_chromatogram.csv">predicted chromatogram</a>, '
                   '<a href="pool_composition.csv">pool composition</a>, '
                   '<a href="best_recipe.json">best feasible recipe</a>' if best is not None else '')
                + '.</p>')
    html_path.write_text(f'<!doctype html><html><head><base href="{html.escape(folder.resolve().as_uri(), quote=True)}/"></head><body>{summary}</body></html>', encoding="utf-8")
    return report
