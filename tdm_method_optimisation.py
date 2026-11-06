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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from tdm_minimal_io import active_components, normalize_config, validate_config
from tdm_minimal_model import simulate, _svg_plot


VARIABLES = ("loading_g_L", "flow_mL_min", "wash_CV", "gradient_start_B",
             "gradient_end_B", "gradient_CV", "pool_start_CV", "pool_end_CV")


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
    gradient = next(stage for stage in p["elution_steps"][:int(p["elution_count"])]
                    if stage["mode"] == "LINEAR" and stage["chemistry_control"] == "BUFFER_B_PERCENT")
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


def _elution_start(result: dict[str, Any]) -> float:
    program = result["simulation"]["program"]
    return float(next(program.stage_start_CV[i] for i, stage in enumerate(program.stages)
                      if stage.kind == "ELUTION" and stage.mode == "LINEAR"))


def _available_pool_span(config: dict[str, Any]) -> float:
    process = config["process"]
    steps = process["elution_steps"][:int(process["elution_count"])]
    first = next(i for i, stage in enumerate(steps)
                 if stage["mode"] == "LINEAR" and stage["chemistry_control"] == "BUFFER_B_PERCENT")
    return sum(float(stage["CV"]) for stage in steps[first:])


def _metrics(result: dict[str, Any], config: dict[str, Any], values: dict[str, float],
             limits: dict[str, Any]) -> dict[str, Any]:
    start = _elution_start(result) + values["pool_start_CV"]
    end = _elution_start(result) + values["pool_end_CV"]
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
               "pool_order_CV": values["pool_end_CV"] - values["pool_start_CV"] - 1e-4}
    for i, item in enumerate(species[1:], start=1):
        name = str(item["name"])
        cap = limits.get("max_impurity_percent", {}).get(name)
        if cap is not None:
            margins[f"maximum_{name}_percent"] = float(cap) - fractions[name]
    if pressure is not None:
        margins["maximum_pressure_bar"] = float(limits["max_pressure_bar"]) - pressure
    return {"yield_percent": yield_pct, "purity_percent": purity, "masses_g": masses.tolist(),
            "fractions_percent": fractions, "loaded_product_g": loaded,
            "pool_start_CV": start, "pool_end_CV": end,
            "pool_volume_mL": (end-start)*float(config["column"]["volume_mL"]),
            "estimated_pressure_bar": pressure, "margins": margins,
            "feasible": bool(all(value >= -1e-6 for value in margins.values()))}


def _check_options(config: dict[str, Any], options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    validation_errors = validate_config(config, strict=True)
    if validation_errors:
        raise ValueError("Complete and validate the selected run before method optimisation: " + "; ".join(validation_errors))
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
    for key in ("min_purity_percent", "max_pool_volume_mL"):
        if not math.isfinite(float(options[key])) or float(options[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    if not 0 < float(options["min_purity_percent"]) <= 100:
        raise ValueError("Minimum purity must be at most 100%.")
    for cap in options.get("max_impurity_percent", {}).values():
        if not 0 <= float(cap) <= 100:
            raise ValueError("Impurity caps must be between 0 and 100%.")
    if options.get("max_pressure_bar") is not None:
        for key in ("max_pressure_bar", "reference_pressure_bar", "reference_flow_mL_min"):
            if options.get(key) is None or not math.isfinite(float(options[key])) or float(options[key]) <= 0:
                raise ValueError("A pressure cap requires positive measured reference pressure and reference flow. The pressure estimate scales linearly with flow for this column and liquid.")
    if not 2 <= int(options.get("max_evaluations", 30)) <= 1000:
        raise ValueError("Method evaluation cap must be 2–1000.")
    timeout = float(options.get("max_trial_minutes", 30))
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
    fields = ["candidate", "model_evaluation", "phase", *VARIABLES, "yield_percent", "purity_percent",
              "feasible", "error", "margins_json"]
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    solve_count = 0
    cap = int(options.get("max_evaluations", 30))
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def evaluate(unit: np.ndarray, phase: str) -> dict[str, Any]:
            nonlocal best, solve_count
            unit = np.clip(np.asarray(unit, dtype=float), 0, 1)
            key = tuple(np.round(unit, 9))
            if key in cache:
                return cache[key]
            values = dict(zip(VARIABLES, (lower + unit * (upper - lower)).tolist()))
            entry: dict[str, Any] = {"values": values, "metrics": None, "error": None, "result": None}
            simulated = False
            try:
                if values["pool_end_CV"] <= values["pool_start_CV"]:
                    raise ValueError("Pool end precedes pool start")
                candidate = _recipe(config, values)
                if values["pool_end_CV"] > _available_pool_span(candidate) + 1e-9:
                    raise ValueError("Pool end lies beyond the selected recipe; skipped without a simulation")
                if solve_count >= cap:
                    raise _Budget()
                solve_count += 1
                simulated = True
                deadline = (time.monotonic() + 60 * float(options.get("max_trial_minutes", 30))
                            if float(options.get("max_trial_minutes", 30)) > 0 else None)
                result = (simulate(candidate, solver_deadline=deadline)
                          if simulator is simulate else simulator(candidate))
                metrics = _metrics(result, candidate, values, options)
                entry.update(metrics=metrics, result=result)
                if metrics["feasible"] and (best is None or metrics["yield_percent"] > best["metrics"]["yield_percent"]):
                    best = entry
            except _Budget:
                raise
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            cache[key] = entry
            metrics = entry["metrics"] or {}
            writer.writerow({"candidate": len(cache), "model_evaluation": solve_count if simulated else "",
                             "phase": phase, **values,
                             "yield_percent": metrics.get("yield_percent", ""),
                             "purity_percent": metrics.get("purity_percent", ""),
                             "feasible": metrics.get("feasible", False),
                             "error": entry["error"] or "",
                             "margins_json": json.dumps(metrics.get("margins", {}))})
            handle.flush()
            return entry

        def score(unit: np.ndarray) -> float:
            e = evaluate(unit, "local")
            m = e["metrics"]
            if m is None:
                return 1e5
            penalty = sum(max(0.0, -float(v))**2 for v in m["margins"].values())
            return -float(m["yield_percent"]) + 1e3 * penalty

        constraint_names = ["minimum_product_purity_percent", "maximum_pool_volume_mL", "pool_order_CV"]
        constraint_names += [f"maximum_{name}_percent" for name in options.get("max_impurity_percent", {})]
        if options.get("max_pressure_bar") is not None:
            constraint_names.append("maximum_pressure_bar")

        def constraint(unit: np.ndarray, name: str) -> float:
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
        evaluate(baseline_unit, "baseline")
        global_count = min(max(2, cap // 2), 32)
        design = qmc.LatinHypercube(d=len(VARIABLES), seed=73).random(global_count)
        budget_hit = False
        local_status = []
        for point in design:
            try:
                evaluate(point, "global")
            except _Budget:
                budget_hit = True
                break
        seeds = sorted(((v["metrics"]["yield_percent"] - 1000*sum(max(0.0,-m)**2 for m in v["metrics"]["margins"].values()),
                         np.array(key)) for key, v in cache.items() if v["metrics"] is not None),
                       key=lambda pair: pair[0], reverse=True)[:3]
        for _, seed in seeds:
            if solve_count >= cap:
                budget_hit = True
                break
            try:
                fit = minimize(score, seed, method="SLSQP", bounds=[(0, 1)]*len(VARIABLES),
                               constraints=[{"type": "ineq", "fun": constraint, "args": (name,)}
                                            for name in constraint_names],
                               options={"maxiter": 12, "ftol": 1e-4, "eps": 1e-4})
                local_status.append({"success": bool(fit.success), "message": str(fit.message)})
            except _Budget:
                budget_hit = True
                break

    report: dict[str, Any] = {"run_name": run_name, "model": config["model"],
        "evaluations": solve_count, "candidates": len(cache),
        "evaluation_cap": cap, "evaluation_cap_reached": budget_hit or solve_count >= cap,
        "local_search": local_status, "best_feasible": best is not None,
        "history_csv": str(history_path), "folder": str(folder),
        "pressure_basis": ("Linear flow-scaled estimate from measured reference pressure; verify against column pressure data."
                           if options.get("max_pressure_bar") is not None else
                           "Pressure was not constrained because no measured reference pressure and flow were supplied."),
        "guarantee": "Best feasible candidate found; global optimality is not established."}
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
        for name, scale in (("pH", .1), ("loading", .05), ("flow", .05),
                            ("salt", .05), ("pool_start", .1), ("pool_end", .1)):
            for sign in (-1, 1):
                varied = dict(values)
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
                    varied_result = (result if varied_config is None else
                                     (simulate(varied_config, solver_deadline=(time.monotonic() + 60 * float(options.get("max_trial_minutes", 30))
                                                                         if float(options.get("max_trial_minutes", 30)) > 0 else None))
                                      if simulator is simulate else simulator(varied_config)))
                    varied_metrics = _metrics(varied_result, varied_config or _recipe(config, varied), varied, options)
                    robustness.append({"variation": name, "direction": sign, "magnitude": scale,
                                       "yield_percent": varied_metrics["yield_percent"],
                                       "purity_percent": varied_metrics["purity_percent"],
                                       "feasible": varied_metrics["feasible"]})
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    robustness.append({"variation": name, "direction": sign, "magnitude": scale,
                                       "error": f"{type(exc).__name__}: {exc}"})
        report["robustness"] = robustness
        with (folder / "pool_composition.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["species", "mass_g", "mass_percent"])
            for i, name in enumerate(names):
                writer.writerow([name, metrics["masses_g"][i], metrics["fractions_percent"][name]])
    report_path = folder / "method_optimisation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    summary = (f"<h1>Method optimisation — {html.escape(run_name)}</h1>"
               f"<p>{html.escape(report['guarantee'])} {report['evaluations']} simulations; "
               f"cap {cap}; cap reached: {report['evaluation_cap_reached']}. "
               f"Candidates checked: {len(cache)}. {html.escape(report['pressure_basis'])}</p>"
               f"<p>Local SLSQP status: {html.escape(json.dumps(local_status))}. "
               "A feasible candidate is reported independently of the local convergence flag.</p>")
    if best is not None:
        summary += (f"<p>Best feasible product yield: {metrics['yield_percent']:.4g}%; "
                    f"purity: {metrics['purity_percent']:.4g}%; pool "
                    f"{metrics['pool_start_CV']:.4g}–{metrics['pool_end_CV']:.4g} CV.</p>"
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
    summary += "<p>See evaluation_history.csv, method_optimisation_report.json and robustness entries for details.</p>"
    html_path = folder / "method_optimisation_report.html"
    html_path.write_text(f"<!doctype html><html><body>{summary}</body></html>", encoding="utf-8")
    report["report_html"] = str(html_path)
    return report
