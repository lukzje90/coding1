from __future__ import annotations

"""Least-squares refinement for the minimal four-mode chromatography model.

Two objective modes are intentionally supported:
1. CHROMATOGRAM: ordinary point-by-point least-squares against an attached raw
   chromatogram CSV.
2. CHROMATOGRAM_AND_COMPOSITION: the same chromatogram objective plus
   species mass-percent observations at user-specified column volumes (CV).

UV predictions use the target and species response factors; only eligible
unlocked factors are refined. Concentration-valued columns are compared
directly and do not refine detector response. RAW_SSE defaults to literal
pointwise equation-9 residuals without inferred baseline or normalization.
NORMALIZED_MSE and initial-median baseline correction are explicit options.
Optional composition observations add normalized composition residuals.
"""

from dataclasses import dataclass
import codecs
import copy
import csv
import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

from tdm_minimal_io import (
    CONFIG_PATH,
    MODEL_EDM_CPA,
    MODEL_EDM_LANGMUIR,
    MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR,
    active_components,
    canonical_fit_parameter_path,
    normalize_config,
    required_parameter_paths,
    save_config,
    selected_components,
    validate_config,
    is_hic,
)
from tdm_minimal_model import simulate, _build_program, _svg_plot, _stage_receipt

PROJECT_DIR = Path(__file__).resolve().parent
ATTACHED_CHROMATOGRAM_CSV = PROJECT_DIR / "least_squares_reference_chromatogram.csv"
FIT_RESULTS_CSV = PROJECT_DIR / "least_squares_fit_results.csv"
FIT_TRACE_CSV = PROJECT_DIR / "least_squares_fit_trace.csv"
FIT_COMPOSITION_CSV = PROJECT_DIR / "least_squares_fit_composition.csv"
FIT_REPORT_HTML = PROJECT_DIR / "least_squares_fit_report.html"
FIT_CONFIG_JSON = PROJECT_DIR / "least_squares_fitted_inputs.json"
FIT_APPLY_JSL = PROJECT_DIR / "least_squares_apply_to_ui.jsl"
FIT_SETTINGS_JSON = PROJECT_DIR / "least_squares_settings.json"
FIT_HISTORY_CSV = PROJECT_DIR / "least_squares_objective_history.csv"
FIT_IMPROVED_CSV = PROJECT_DIR / "least_squares_improved_parameters.csv"
FIT_IMPROVED_TXT = PROJECT_DIR / "least_squares_improved_parameters.txt"

MODE_CHROMATOGRAM = "CHROMATOGRAM"
MODE_CHROMATOGRAM_AND_COMPOSITION = "CHROMATOGRAM_AND_COMPOSITION"
SUPPORTED_MODES = {MODE_CHROMATOGRAM, MODE_CHROMATOGRAM_AND_COMPOSITION}
MAX_COMPOSITION_POINTS = 10
SAMPLE_ORDER_FIELD = "__sample_order__"


@dataclass(frozen=True)
class ChromatogramReference:
    cv: np.ndarray
    signal: np.ndarray
    x_column: str
    signal_column: str
    x_unit: str
    source_path: str
    x_axis_assumption: str | None = None
    point_weights: np.ndarray | None = None
    weight_column: str | None = None


@dataclass(frozen=True)
class FitParameter:
    path: str
    label: str
    transform: str  # LOG10 or LINEAR
    lower: float
    upper: float
    initial: float


def _norm_header(text: str) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def _parse_numeric_cell(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"na", "n/a", "nan", "null", "none", "--", "-"}:
        return None
    text = text.replace("\u00a0", "").replace("\u202f", "").replace(" ", "").replace("−", "-")
    # Instrument exports often store elapsed time as mm:ss or hh:mm:ss.
    duration = re.fullmatch(r"([+-]?\d+):([0-5]?\d)(?::([0-5]?\d(?:[.,]\d+)?))?", text)
    if duration:
        first = float(duration.group(1))
        second = float(duration.group(2))
        third = duration.group(3)
        if third is None:
            return first * 60.0 + second
        return first * 3600.0 + second * 60.0 + float(third.replace(",", "."))
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        left, right = text.rsplit(",", 1)
        if right.isdigit() and 1 <= len(right) <= 6:
            text = left.replace(",", "") + "." + right
        else:
            text = text.replace(",", "")
    try:
        out = float(text)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _is_number(value: Any) -> bool:
    return _parse_numeric_cell(value) is not None


def _parse_path(path: str) -> list[str | int]:
    import re
    out: list[str | int] = []
    for name, index in re.findall(r"([^\.\[\]]+)|\[(\d+)\]", path):
        out.append(name if name else int(index))
    return out


def _get_path(data: Any, path: str) -> Any:
    cur = data
    for token in _parse_path(path):
        cur = cur[token]
    return cur


def _set_path(data: Any, path: str, value: Any) -> None:
    tokens = _parse_path(path)
    cur = data
    for token in tokens[:-1]:
        cur = cur[token]
    cur[tokens[-1]] = value


def _normalize_chromatogram_path_text(path: Path | str) -> str:
    """Clean file-picker text, including JMP's leading slash before Windows drives."""
    source_text = str(path).strip().strip('"').strip("'")
    if not source_text:
        raise ValueError("Chromatogram file path cannot be blank.")
    # JMP may return its POSIX-style Windows path (for example, /C:/data/run.csv)
    # or a slash-converted variant (\\C:\\data\\run.csv). Strip only the
    # separator(s) immediately before a drive letter; preserve UNC paths.
    return re.sub(r"^[\\/]+(?=[A-Za-z]:[\\/])", "", source_text)


def _resolve_chromatogram_path(path: Path | str) -> Path:
    """Resolve a chromatogram CSV/Excel file, including JMP path variants."""
    received_text = str(path).strip().strip('"').strip("'")
    source_text = _normalize_chromatogram_path_text(path)
    candidate = Path(source_text).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.suffix:
        for extension in (".csv", ".CSV", ".xlsx", ".xlsm"):
            with_extension = Path(str(candidate) + extension)
            if with_extension.is_file():
                return with_extension.resolve()
        raise FileNotFoundError(
            "Chromatogram CSV/Excel file not found. Checked the entered path and "
            f"common extensions: {candidate}. In JMP, click Attach / validate chromatogram "
            "to choose the file."
        )
    raise FileNotFoundError(
        "Chromatogram CSV not found after JMP path normalization. "
        f"Path received: {received_text!r}. Path checked: {candidate}. "
        "Check the full filename and extension, or choose the file with the JMP picker."
    )


def _read_numeric_rows(path: Path, *, sheet_name: str | None = None) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Chromatogram file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        # Keep the Excel dependency optional for users fitting CSV exports only.
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise RuntimeError("Excel chromatograms require openpyxl (pip install -r requirements.txt).") from exc
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            best = None
            sheets = workbook.worksheets
            if sheet_name:
                sheets = [sheet for sheet in sheets if sheet.title.strip().casefold() == sheet_name.strip().casefold()]
                if not sheets:
                    raise ValueError(f"Excel worksheet {sheet_name!r} not found. Available sheets: {', '.join(workbook.sheetnames)}")
            for sheet in sheets:
                records = []
                for row in sheet.iter_rows(values_only=True):
                    if any(cell is not None for cell in row):
                        records.append([str(c) if c is not None else "" for c in row])
                if not records:
                    continue
                buffer = io.StringIO()
                csv.writer(buffer).writerows(records)
                try:
                    fields, rows = _read_numeric_rows_text(buffer.getvalue())
                    x, y, _unit = _detect_columns(fields, rows)
                    count = sum(_is_number(r.get(y)) and (x == SAMPLE_ORDER_FIELD or _is_number(r.get(x))) for r in rows)
                except ValueError:
                    continue
                if best is None or count > best[0]:
                    best = (count, fields, rows)
            if best is None:
                raise ValueError("No worksheet has at least three numeric chromatogram samples; use headered CV/time and signal columns.")
            return best[1], best[2]
        finally:
            workbook.close()
    if path.suffix.lower() == ".xls":
        raise ValueError("Legacy .xls is not supported. Save the Excel workbook as .xlsx or CSV.")
    if sheet_name:
        raise ValueError("Worksheet selection can only be used with Excel .xlsx/.xlsm files, not CSV.")
    with path.open("rb") as fh:
        prefix = fh.read(4)
    if prefix.startswith(codecs.BOM_UTF32_LE) or prefix.startswith(codecs.BOM_UTF32_BE):
        encodings = ("utf-32",)
    elif prefix.startswith(codecs.BOM_UTF16_LE) or prefix.startswith(codecs.BOM_UTF16_BE):
        encodings = ("utf-16",)
    elif prefix.startswith(codecs.BOM_UTF8):
        encodings = ("utf-8-sig",)
    else:
        # UTF-8 is preferred. CSVs exported by Windows applications may instead
        # use the active Windows code page, so retry those common encodings.
        encodings = ("utf-8-sig", "cp1252", "latin-1")

    last_decode_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
    else:
        raise ValueError(
            "Chromatogram CSV text encoding could not be decoded. Supported formats "
            "include UTF-8, UTF-16/UTF-32 with a byte-order mark, and Windows-1252."
        ) from last_decode_error

    return _read_numeric_rows_text(text)


def _read_numeric_rows_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    delimiters: list[str] = []
    try:
        guessed = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
        delimiters.append(guessed)
    except csv.Error:
        pass
    delimiters.extend(delim for delim in (",", ";", "\t", "|") if delim not in delimiters)

    best: tuple[tuple[int, int, int, int], list[str], list[dict[str, str]]] | None = None
    for delimiter in delimiters:
        try:
            records = list(csv.reader(io.StringIO(text), delimiter=delimiter, skipinitialspace=True))
        except csv.Error:
            continue
        records = [[str(cell).strip().lstrip("\ufeff") for cell in record] for record in records]
        records = [record for record in records if any(cell for cell in record)]
        for header_index, raw_header in enumerate(records[:40]):
            if not raw_header or not any(raw_header):
                continue
            fields = [cell or f"Column_{i + 1}" for i, cell in enumerate(raw_header)]
            # Make repeated/blank instrument column names safe for DictReader-like use.
            seen: dict[str, int] = {}
            unique_fields: list[str] = []
            for i, field in enumerate(fields):
                count = seen.get(field, 0) + 1
                seen[field] = count
                unique_fields.append(field if count == 1 else f"{field}_{count}")
            data_records = records[header_index + 1:]
            pair_rows = 0
            numeric_cells = 0
            single_numeric_rows = 0
            for record in data_records:
                values = [_parse_numeric_cell(record[i] if i < len(record) else "") for i in range(len(fields))]
                count = sum(value is not None for value in values)
                numeric_cells += count
                if count >= 2:
                    pair_rows += 1
                elif count == 1:
                    single_numeric_rows += 1
            if len(fields) > 1:
                usable_rows = pair_rows
            else:
                usable_rows = single_numeric_rows
            if usable_rows < 3:
                continue
            normalized_header = " ".join(_norm_header(field) for field in unique_fields)
            header_quality = sum(
                token in normalized_header
                for token in ("time", "volume", "cv", "retention", "sample", "index", "uv", "signal", "absorbance", "detector", "response", "concentration")
            )
            score = (usable_rows, numeric_cells, header_quality, header_index)
            data: list[dict[str, str]] = []
            for record in data_records:
                data.append({field: record[i].strip() if i < len(record) else "" for i, field in enumerate(unique_fields)})
            if best is None or score > best[0]:
                best = (score, unique_fields, data)

        # Headerless numeric exports are common in detector-only files. Preserve
        # the samples and synthesize column names if the first record is numeric.
        if records and all(_parse_numeric_cell(cell) is not None for cell in records[0]):
            width = len(records[0])
            fields = [f"Column_{i + 1}" for i in range(width)]
            data = [{field: record[i].strip() if i < len(record) else "" for i, field in enumerate(fields)} for record in records]
            usable = sum(sum(_parse_numeric_cell(row[field]) is not None for field in fields) >= min(2, width) for row in data)
            if usable >= 3:
                score = (usable, usable * width, 0, -1)
                if best is None or score > best[0]:
                    best = (score, fields, data)

    if best is None:
        raise ValueError(
            "Could not read at least three chromatogram samples from this CSV. Tried comma, "
            "semicolon, tab, and pipe separators, including instrument metadata rows. "
            "Export a table with a header and numeric detector values; a numeric time/CV/volume "
            "column is preferred, but a single detector column can use sample order."
        )
    _score, fields, rows = best
    if len(rows) < 3:
        raise ValueError("Chromatogram CSV must contain at least three data rows.")
    return fields, rows


def _column_numeric_count(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for r in rows if _is_number(r.get(field)))


def _detect_columns(fields: list[str], rows: list[dict[str, str]]) -> tuple[str, str, str]:
    by_norm = {_norm_header(f): f for f in fields}
    x_aliases: list[tuple[str, str]] = [
        ("cv", "CV"), ("columnvolume", "CV"), ("columnvolumes", "CV"),
        ("columnvolumecv", "CV"), ("elutionvolumecv", "CV"),
        ("columnvolumeml", "ML"), ("columnvolumesml", "ML"), ("columnvolumeinml", "ML"),
        ("elutionvolumeml", "ML"), ("retentionvolumeml", "ML"), ("volumeml", "ML"),
        ("elutionvolumeinml", "ML"), ("elutionvolume", "ML"), ("retentionvolume", "ML"), ("volume", "ML"), ("ml", "ML"),
        ("retentiontimemin", "MIN"), ("acquisitiontimemin", "MIN"), ("samplingtimemin", "MIN"), ("retentiontime", "MIN"), ("acquisitiontime", "MIN"), ("samplingtime", "MIN"), ("elapsedtimemin", "MIN"), ("elapsedtime", "MIN"),
        ("elutiontimemin", "MIN"), ("timemin", "MIN"), ("minutes", "MIN"), ("minute", "MIN"), ("min", "MIN"), ("time", "MIN"),
        ("retentiontimes", "S"), ("acquisitiontimes", "S"), ("samplingtimes", "S"), ("elapsedtimes", "S"), ("elutiontimes", "S"),
        ("timesec", "S"), ("timesecs", "S"), ("times", "S"), ("timeinseconds", "S"), ("seconds", "S"), ("second", "S"), ("sec", "S"),
        ("timeh", "H"), ("timehr", "H"), ("timehrs", "H"), ("timehours", "H"), ("hours", "H"),
        ("sample", "INDEX"), ("sampleid", "INDEX"), ("samplenumber", "INDEX"), ("sampleindex", "INDEX"),
        ("datapoint", "INDEX"), ("point", "INDEX"), ("frame", "INDEX"), ("sequence", "INDEX"),
        ("acquisitionnumber", "INDEX"), ("index", "INDEX"), ("scan", "INDEX"), ("scannumber", "INDEX"),
        ("x", "CV"),
    ]
    numeric_fields = [f for f in fields if _column_numeric_count(rows, f) >= 3]
    # A point-weight column is not a detector response or x-axis.
    numeric_fields = [f for f in numeric_fields if _norm_header(f) not in {"weight", "weights", "pointweight", "sampleweight", "lsqweight", "fitweight"}]
    # Prefer an explicitly labelled UV/mAU channel to generic "Signal" when
    # an instrument export contains several numerical detector channels.
    signal_aliases = ["uvmau", "uv280mau", "uv280", "uv", "absorbancemau"]
    signal_tags = ("uv", "a280", "mau", "absorbance")
    signal_field = ""
    for alias in signal_aliases:
        candidate = by_norm.get(alias)
        if candidate in numeric_fields:
            signal_field = candidate
            break
    if not signal_field:
        for field in fields:
            norm = _norm_header(field)
            if field in numeric_fields and any(tag in norm for tag in signal_tags):
                signal_field = field
                break
    if not signal_field:
        for alias in ("absorbance", "signal", "y", "totalproteingl", "totalgl", "proteingl", "concentrationgl", "response"):
            candidate = by_norm.get(alias)
            if candidate in numeric_fields:
                signal_field = candidate
                break
    if not signal_field:
        for field in fields:
            norm = _norm_header(field)
            if field in numeric_fields and any(tag in norm for tag in ("signal", "detector", "response", "intensity", "totalprotein", "concentrationgl", "yaxis")):
                signal_field = field
                break

    x_field = ""
    x_unit = "CV"
    for alias, unit in x_aliases:
        candidate = by_norm.get(alias)
        if candidate and candidate in numeric_fields and candidate != signal_field:
            x_field = candidate
            x_unit = unit
            break
    if not x_field:
        x_candidates = [f for f in numeric_fields if f != signal_field]
        if x_candidates:
            x_field = x_candidates[0]
            x_unit = "CV"
    if x_field and x_field != SAMPLE_ORDER_FIELD and x_unit == "MIN":
        if any(":" in str(row.get(x_field) or "") for row in rows):
            x_unit = "S"
    if not signal_field:
        candidates = [f for f in numeric_fields if f != x_field]
        if candidates:
            signal_field = candidates[0]
    if not signal_field:
        raise ValueError(
            "No numeric detector/signal column could be detected. Keep the detector values "
            "in a numeric CSV column (for example UV 280 [mAU] or Signal)."
        )
    if not x_field:
        # A single detector column is still usable when sampling was uniform. Its
        # row order is mapped across the selected recipe's full process duration.
        x_field = SAMPLE_ORDER_FIELD
        x_unit = "INDEX"
    return x_field, signal_field, x_unit


def _select_columns(fields: list[str], rows: list[dict[str, str]], *,
                    x_column: str | None = None, signal_column: str | None = None,
                    x_unit: str | None = None) -> tuple[str, str, str]:
    auto_x, auto_signal, auto_unit = _detect_columns(fields, rows)
    by_header = {_norm_header(f): f for f in fields}

    def choose(requested: str | None, inferred: str, role: str) -> str:
        if not str(requested or "").strip():
            return inferred
        selected = by_header.get(_norm_header(str(requested)))
        if selected is None:
            raise ValueError(f"Requested {role} column {requested!r} does not exist. Available: {fields}")
        if _column_numeric_count(rows, selected) < 3:
            raise ValueError(f"Requested {role} column {requested!r} does not contain at least three numeric samples.")
        return selected

    x = choose(x_column, auto_x, "x-axis")
    y = choose(signal_column, auto_signal, "signal")
    if x == y:
        raise ValueError("Reference x-axis and signal must be different columns.")
    unit = str(x_unit or "AUTO").strip().upper()
    if unit == "AUTO":
        if x != auto_x:
            raise ValueError("When overriding the automatically detected x-axis, also set the reference X-axis unit explicitly.")
        unit = auto_unit
    if unit not in {"CV", "ML", "MIN", "S", "H", "INDEX"}:
        raise ValueError("Reference x-axis unit must be AUTO, CV, ML, MIN, S, H, or INDEX.")
    return x, y, unit


def inspect_chromatogram_csv(path: Path | str, *, sheet_name: str | None = None,
                             x_column: str | None = None, signal_column: str | None = None,
                             x_unit: str | None = None, x_origin: str | None = None) -> dict[str, Any]:
    p = _resolve_chromatogram_path(path)
    fields, rows = _read_numeric_rows(p, sheet_name=sheet_name)
    x_field, signal_field, unit = _select_columns(fields, rows, x_column=x_column, signal_column=signal_column, x_unit=x_unit)
    origin = _validate_x_origin(x_origin, unit)
    paired = (
        _column_numeric_count(rows, signal_field)
        if x_field == SAMPLE_ORDER_FIELD else
        sum(1 for r in rows if _is_number(r.get(x_field)) and _is_number(r.get(signal_field)))
    )
    if paired < 3:
        raise ValueError("Detected chromatogram columns do not contain at least three paired numeric points.")
    assumption = None
    if x_field == SAMPLE_ORDER_FIELD:
        assumption = "No numeric time/CV/volume axis was present; sample order is assumed uniformly spaced across this profile's full process duration."
    return {
        "path": str(p),
        "rows": paired,
        "x_column": "sample order (assumed)" if x_field == SAMPLE_ORDER_FIELD else x_field,
        "signal_column": signal_field,
        "x_unit": unit,
        "x_origin": origin,
        "sheet_name": sheet_name or "auto",
        "columns": fields,
        "x_axis_assumption": assumption,
        "weight_column": _weight_column(fields),
    }


def attach_chromatogram_csv(source: Path | str, **column_options: Any) -> dict[str, Any]:
    source_path = _resolve_chromatogram_path(source)
    info = inspect_chromatogram_csv(source_path, **column_options)
    # Profile paths are saved directly in their independent LSQ process table.
    # Never copy Excel bytes to a .csv filename or overwrite another run's data.
    info["source_path"] = str(source_path)
    return info


def _weight_column(fields: list[str]) -> str | None:
    return next((f for f in fields if _norm_header(f) in {
        "weight", "weights", "pointweight", "sampleweight", "lsqweight", "fitweight"}), None)


def _x_to_cv(x: np.ndarray, x_unit: str, config: dict[str, Any]) -> np.ndarray:
    unit = str(x_unit).upper()
    if unit == "CV":
        return x
    V = float(config["column"]["volume_mL"])
    if unit == "ML":
        return x / V
    if unit in {"MIN", "S", "H", "INDEX"}:
        # Flow may change between load, PLW and elution stages.  Convert time
        # using the exact same piecewise program as the mechanistic solver.
        program = _build_program(config, np.zeros(max(1, len(active_components(config))), dtype=float))
        values = np.asarray(x, dtype=float)
        if unit == "INDEX":
            span = float(np.max(values) - np.min(values))
            fraction = np.zeros_like(values) if span <= 0 else (values - float(np.min(values))) / span
            seconds = fraction * program.total_time_s
        else:
            seconds = values * (3600.0 if unit == "H" else 60.0 if unit == "MIN" else 1.0)
        return np.asarray([
            float(t) * program.stages[0].flow_mL_min / (60.0 * V) if t < 0.0 else
            program.total_CV + (float(t) - program.total_time_s) * program.stages[-1].flow_mL_min / (60.0 * V) if t > program.total_time_s else
            program.time_to_cv(float(t)) for t in seconds
        ], dtype=float)
    raise ValueError(f"Unsupported chromatogram x-axis unit: {x_unit}")


def _validate_x_origin(x_origin: str | None, x_unit: str) -> str:
    origin = str(x_origin or "RUN_START").strip().upper()
    if origin not in {"RUN_START", "ELUTION_START"}:
        raise ValueError("Reference X-axis zero must be RUN_START or ELUTION_START.")
    if origin == "ELUTION_START" and x_unit not in {"CV", "ML"}:
        raise ValueError("Elution-start X-axis zero requires CV or mL units.")
    return origin


def _elution_start_cv(config: dict[str, Any]) -> float:
    program = _build_program(config, np.zeros(max(1, len(active_components(config))), dtype=float))
    for index, stage in enumerate(program.stages):
        if stage.kind == "ELUTION":
            return float(program.stage_start_CV[index])
    raise ValueError("The selected LSQ process recipe has no elution stage; elution-start X-axis zero cannot be used.")


def load_chromatogram_reference(path: Path | str, config: dict[str, Any], *,
                                sheet_name: str | None = None, x_column: str | None = None,
                                signal_column: str | None = None, x_unit: str | None = None,
                                x_origin: str | None = None) -> ChromatogramReference:
    p = _resolve_chromatogram_path(path)
    fields, rows = _read_numeric_rows(p, sheet_name=sheet_name)
    x_field, signal_field, unit = _select_columns(fields, rows, x_column=x_column, signal_column=signal_column, x_unit=x_unit)
    origin = _validate_x_origin(x_origin, unit)
    weights_field = _weight_column(fields)
    pairs: list[tuple[float, float, float]] = []
    if x_field == SAMPLE_ORDER_FIELD:
        for row in rows:
            signal = _parse_numeric_cell(row.get(signal_field))
            if signal is not None:
                weight = _reference_row_weight(row, weights_field)
                pairs.append((float(len(pairs)), signal, weight))
    else:
        for row in rows:
            x_value = _parse_numeric_cell(row.get(x_field))
            signal = _parse_numeric_cell(row.get(signal_field))
            if x_value is not None and signal is not None:
                weight = _reference_row_weight(row, weights_field)
                pairs.append((x_value, signal, weight))
    if len(pairs) < 3:
        raise ValueError("Chromatogram CSV contains fewer than three paired numeric points.")
    arr = np.asarray(pairs, dtype=float)
    order = np.argsort(arr[:, 0])
    x = arr[order, 0]
    y = arr[order, 1]
    weights = arr[order, 2]
    cv = _x_to_cv(x, unit, config)
    if origin == "ELUTION_START":
        cv = cv + _elution_start_cv(config)
    finite = np.isfinite(cv) & np.isfinite(y)
    cv, y, weights = cv[finite], y[finite], weights[finite]
    if len(cv) < 3:
        raise ValueError("Chromatogram CSV contains fewer than three finite points after conversion to CV.")
    # Repeated reference coordinates are retained: every measured sample is
    # one equation-9 residual, including replicates at the same time/CV.
    display_x_field = "sample order (assumed)" if x_field == SAMPLE_ORDER_FIELD else x_field
    assumption = None
    if x_field == SAMPLE_ORDER_FIELD:
        assumption = "No numeric time/CV/volume axis was present; sample order is assumed uniformly spaced across the selected profile's process duration."
    elif unit == "INDEX":
        assumption = "The numeric sample/point index is mapped uniformly across the selected profile's process duration."
    if not np.any(weights > 0):
        raise ValueError("All reference sample weights are zero; at least one must be positive.")
    return ChromatogramReference(cv=cv, signal=y, x_column=display_x_field, signal_column=signal_field, x_unit=unit, source_path=str(p), x_axis_assumption=assumption, point_weights=weights, weight_column=weights_field)


def _reference_row_weight(row: dict[str, str], weight_field: str | None) -> float:
    if weight_field is None:
        return 1.0
    weight = _parse_numeric_cell(row.get(weight_field))
    if weight is None or weight < 0 or not math.isfinite(weight):
        raise ValueError(f"Point-weight column {weight_field!r} must contain finite nonnegative values for every paired sample.")
    return weight


def _positive_log_spec(path: str, label: str, value: float, decades: float = 1.5) -> FitParameter:
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{label} must be positive before least-squares refinement.")
    x = math.log10(value)
    return FitParameter(path, label, "LOG10", x - decades, x + decades, x)


def _is_concentration_signal(signal_column: str) -> bool:
    signal_norm = _norm_header(signal_column)
    return any(tag in signal_norm for tag in ("totalproteingl", "totalgl", "proteingl", "concentrationgl"))


def _linear_z_spec(path: str, label: str, value: float) -> FitParameter:
    span = max(25.0, abs(value) * 1.5)
    return FitParameter(path, label, "LINEAR", value - span, value + span, value)


def _default_unlocked_parameter_paths(config: dict[str, Any], *, fit_uv_response: bool = True) -> set[str]:
    """Keep the established default fit set unless the user changes lock states."""
    config = normalize_config(config)
    model = str(config["model"])
    paths: set[str] = set()
    for i, _row in enumerate(active_components(config)):
        prefix = f"components[{i}]"
        if model in {MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR}:
            paths.update({f"{prefix}.qmax_g_L", f"{prefix}.b_L_g", f"{prefix}.salt_sensitivity_per_M"})
        else:
            paths.update({f"{prefix}.delta_ref", f"{prefix}.As_m_inv"})
            paths.add(f"{prefix}.salt_sensitivity_per_M" if is_hic(config) else f"{prefix}.Z_ref")
            if model == MODEL_TDM_CPA:
                paths.add(f"{prefix}.kkin_star_s")
        if i > 0 and fit_uv_response:
            paths.add(f"{prefix}.uv_response_factor_mAU_L_g")
    return paths


def _all_fit_parameter_specs(config: dict[str, Any], *, fit_uv_response: bool = True,
                             candidate_paths: Iterable[str] | None = None) -> list[FitParameter]:
    """Build bounded specs for every required mechanistic input except column geometry.

    Feed composition, load/flow, buffers, PLWs and elution are experimental
    inputs and remain fixed per profile. Column volume and length are also fixed
    and are never eligible for least-squares fitting.
    """
    config = normalize_config(config)
    model = str(config["model"])
    paths = [canonical_fit_parameter_path(p) for p in candidate_paths] if candidate_paths is not None else required_parameter_paths(config)
    specs: list[FitParameter] = []
    labels = {
        "buffer_dispersion_mL": "Buffer mixer dispersion volume [mL]",
        "salt_dispersion_mm2_s": "Buffer/salt axial dispersion [mm²/s]",
        "uv_dead_volume_mL": "UV system dead volume [mL]",
        "conductivity_dead_volume_mL": "Conductivity system dead volume [mL]",
        "uv_to_protein_mAU_L_g": "Target UV → protein response factor [mAU·L/g]",
        "conductivity_to_salt_M_per_mS_cm": "Conductivity → salt conversion factor [M per (mS/cm)]",
        "bead_radius_um": "Bead radius r_p [µm]",
        "void_fraction": "Void fraction ε_v [-]",
        "particle_porosity": "Particle porosity ε_p [-]",
        "salt_axial_dispersion_mm2_s": "Salt axial dispersion D_ax,salt [mm²/s]",
        "total_bed_porosity": "Total bed porosity ε [-]",
        "ligand_surface_density_umol_m2": "Ligand surface density Γ_L [µmol/m²]",
        "system_specific_adsorption_parameter": "System-specific adsorption parameter [-]",
        "D_ax_mm2_s": "Axial dispersion D_ax,i [mm²/s]",
        "k_eff_um_s": "Effective mass transfer k_eff,i [µm/s]",
        "accessible_particle_porosity": "Accessible particle porosity ε_p,i [-]",
        "D_app_mm2_s": "Apparent dispersion D_app,i [mm²/s]",
        "qmax_g_L": "Saturation capacity qmax,i [g/L stationary phase]",
        "b_L_g": "Competitive Langmuir b_i [L/g]",
        "salt_sensitivity_per_M": "HIC salt sensitivity k_s,i [M⁻¹]",
        "diameter_nm": "Protein diameter [nm]",
        "As_m_inv": "Accessible adsorber surface A_s,i [m⁻¹]",
        "Z_ref": "Protein charge Z_i [-]",
        "delta_ref": "CPA interaction parameter Δ_i [-]",
        "kkin_star_s": "CPA kinetic parameter k*kin,i [s⁻¹]",
        "pH_ref": "CPA reference pH",
        "Z1_per_pH": "CPA Z1,i [-]",
        "Z2_per_pH2": "CPA Z2,i [-]",
        "Z3_per_pH3": "CPA Z3,i [-]",
        "delta_pH_slope_m2_C": "CPA Δ1,i [m²/C]",
        "uv_response_factor_mAU_L_g": "UV → protein response factor [mAU·L/g]",
    }
    for path in paths:
        if path in {"column.volume_mL", "column.length_mm"} or path.endswith(".mass_percent"):
            continue
        if not fit_uv_response and (path.endswith("uv_response_factor_mAU_L_g") or path == "conversion.uv_to_protein_mAU_L_g"):
            continue
        try:
            value = float(_get_path(config, path))
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        field = path.rsplit(".", 1)[-1]
        if field in {"void_fraction", "particle_porosity", "total_bed_porosity", "accessible_particle_porosity"}:
            specs.append(FitParameter(path, labels.get(field, field), "LINEAR", 1e-9, 0.999999,
                                      float(np.clip(value, 1e-9, 0.999999))))
        elif field == "salt_sensitivity_per_M":
            specs.append(FitParameter(path, labels[field], "LINEAR", 0.0 if is_hic(config) else -25.0, 25.0,
                                      float(np.clip(value, 0.0 if is_hic(config) else -25.0, 25.0))))
        elif field in {"buffer_dispersion_mL", "salt_dispersion_mm2_s", "uv_dead_volume_mL", "conductivity_dead_volume_mL", "D_ax_mm2_s", "D_app_mm2_s", "salt_axial_dispersion_mm2_s"}:
            upper = max(10.0, value * 20.0, float(config["column"]["volume_mL"]) * 5.0)
            specs.append(FitParameter(path, labels.get(field, field), "LINEAR", 0.0, upper,
                                      value))
        elif field == "Z_ref":
            specs.append(FitParameter(path, labels[field], "LINEAR", -500.0, 500.0,
                                      float(np.clip(value, -500.0, 500.0))))
        elif field == "pH_ref":
            specs.append(FitParameter(path, labels[field], "LINEAR", 0.0, 14.0,
                                      float(np.clip(value, 0.0, 14.0))))
        elif field in {"Z1_per_pH", "Z2_per_pH2", "Z3_per_pH3", "delta_pH_slope_m2_C"}:
            span = max(1.0, abs(value) * 2.0)
            specs.append(FitParameter(path, labels[field], "LINEAR", value - span, value + span, value))
        elif field in {"uv_to_protein_mAU_L_g", "uv_response_factor_mAU_L_g", "conductivity_to_salt_M_per_mS_cm",
                       "bead_radius_um", "salt_axial_dispersion_mm2_s", "ligand_surface_density_umol_m2",
                       "system_specific_adsorption_parameter", "D_ax_mm2_s", "k_eff_um_s", "D_app_mm2_s",
                       "qmax_g_L", "b_L_g", "diameter_nm", "As_m_inv", "delta_ref", "kkin_star_s"}:
            try:
                specs.append(_positive_log_spec(path, labels.get(field, field), value))
            except ValueError:
                continue
    return specs


def fit_parameter_lock_catalog(config: dict[str, Any]) -> list[FitParameter]:
    """Return all currently required mechanistic parameters that can be unlocked."""
    return _all_fit_parameter_specs(config, fit_uv_response=True)


def fit_parameter_specs(config: dict[str, Any], *, fit_uv_response: bool = True,
                        unlocked_paths: Iterable[str] | None = None,
                        candidate_paths: Iterable[str] | None = None) -> list[FitParameter]:
    """Return the model-backed parameters refined by least squares.

    Eligible instrument, adsorption and column-transport values are refined only
    when unlocked. Recipe inputs, geometry and species composition remain fixed.
    """
    config = normalize_config(config)
    model = str(config["model"])
    if unlocked_paths is not None:
        unlocked = set(canonical_fit_parameter_path(path) for path in unlocked_paths)
        candidates = _all_fit_parameter_specs(config, fit_uv_response=fit_uv_response,
                                              candidate_paths=candidate_paths)
        return [spec for spec in candidates if spec.path in unlocked]
    persisted = config.get("least_squares", {}).get("unlocked_paths")
    unlocked = set(persisted) if persisted is not None else _default_unlocked_parameter_paths(config, fit_uv_response=fit_uv_response)
    return [spec for spec in _all_fit_parameter_specs(config, fit_uv_response=fit_uv_response, candidate_paths=candidate_paths)
            if spec.path in unlocked]


def fit_parameter_text(config: dict[str, Any]) -> str:
    specs = fit_parameter_specs(config)
    return "\n".join(f"• {s.label}" for s in specs)


def _apply_vector(base: dict[str, Any], specs: list[FitParameter], vector: np.ndarray) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    for spec, raw in zip(specs, np.asarray(vector, dtype=float)):
        value = 10.0 ** float(raw) if spec.transform == "LOG10" else float(raw)
        _set_path(cfg, spec.path, value)
    return normalize_config(cfg)


def _encode_initial(specs: list[FitParameter]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.array([s.initial for s in specs], dtype=float)
    lo = np.array([s.lower for s in specs], dtype=float)
    hi = np.array([s.upper for s in specs], dtype=float)
    return x0, lo, hi


def validate_composition_references(
    config: dict[str, Any],
    references: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = [str(r["name"]) for r in active_components(config)]
    out: list[dict[str, Any]] = []
    for k, row in enumerate(references, start=1):
        try:
            cv = float(row.get("CV"))
        except Exception:
            raise ValueError(f"Composition reference {k}: CV is required and must be numeric.")
        if not math.isfinite(cv) or cv < 0:
            raise ValueError(f"Composition reference {k}: CV must be finite and >= 0.")
        values: list[float] = []
        for name in names:
            try:
                value = float(row["mass_percent"][name])
            except Exception:
                raise ValueError(f"Composition reference {k}: mass% for {name} is required.")
            if not math.isfinite(value) or value < 0 or value > 100:
                raise ValueError(f"Composition reference {k}: mass% for {name} must be between 0 and 100.")
            values.append(value)
        total = sum(values)
        if abs(total - 100.0) > 0.05:
            raise ValueError(f"Composition reference {k}: species mass% must sum to 100 (current total {total:g}).")
        out.append({"CV": cv, "mass_percent": dict(zip(names, values))})
    if not out:
        raise ValueError("At least one composition reference CV is required for the chromatogram + mass% option.")
    return out


def _baseline_correct(signal: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(signal, dtype=float)
    n0 = max(3, min(len(y), int(math.ceil(len(y) * 0.05))))
    baseline = float(np.median(y[:n0]))
    corrected = y - baseline
    scale = max(float(np.max(corrected) - np.min(corrected)), float(np.max(np.abs(corrected))), 1e-12)
    return corrected, baseline, scale


def _chromatogram_residuals(config: dict[str, Any], ref: ChromatogramReference, trace: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    sim_cv = np.asarray(trace["CV"], dtype=float)
    sim_total = np.asarray(trace["total_g_L"], dtype=float)
    tolerance = 1e-9 * max(1.0, float(np.max(sim_cv)))
    mask = (ref.cv >= float(np.min(sim_cv)) - tolerance) & (ref.cv <= float(np.max(sim_cv)) + tolerance)
    if not np.all(mask):
        raise ValueError("Reference chromatogram extends outside the simulated recipe. Extend the run/flush duration or crop the reference explicitly; fitting will not silently discard measured points.")
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("Fewer than three chromatogram points overlap the simulated CV range.")
    x = np.clip(ref.cv[mask], float(np.min(sim_cv)), float(np.max(sim_cv)))
    y_meas = ref.signal[mask]
    y_sim = np.interp(x, sim_cv, sim_total)
    _meas_corr, estimated_baseline, measured_scale = _baseline_correct(y_meas)
    options = config.get("least_squares", {})
    baseline = estimated_baseline if options.get("baseline_mode") == "INITIAL_MEDIAN" else float(options.get("detector_baseline", 0.0))
    objective = options.get("objective", "RAW_SSE")
    normalized = objective == "NORMALIZED_MSE"
    scale = measured_scale if normalized else 1.0
    concentration_signal = _is_concentration_signal(ref.signal_column)
    if concentration_signal:
        detector_scale = 1.0
        predicted_zero_baseline = y_sim
        detector_scale_source = "concentration CSV: no UV conversion applied"
    else:
        detector_scale = None
        predicted_zero_baseline = np.interp(x, sim_cv, np.asarray(trace["uv_mAU"], dtype=float))
        detector_scale_source = "species-specific UV response factors (target factor plus fitted impurity factors)"
    predicted_signal = baseline + predicted_zero_baseline
    residual = (predicted_signal - y_meas) / scale
    if normalized:
        residual = residual / math.sqrt(len(residual))
    if objective == "WEIGHTED_RMSE":
        point_weights = ref.point_weights if ref.point_weights is not None else np.ones(len(residual))
        residual = residual * np.sqrt(point_weights / float(np.sum(point_weights)))
    info = {
        "CV": x,
        "measured": y_meas,
        "simulated_total_g_L": y_sim,
        "predicted_detector_signal": predicted_signal,
        "baseline": baseline,
        "detector_scale": detector_scale,
        "detector_scale_source": detector_scale_source,
        "normalization_scale": scale,
        "measured_signal_scale": measured_scale,
        "objective": options.get("objective", "RAW_SSE"),
        "overlap_points": len(x),
        "point_weights": ref.point_weights if ref.point_weights is not None else np.ones(len(x)),
        "point_weight_column": ref.weight_column,
    }
    return residual, info


def _composition_residuals(
    config: dict[str, Any],
    references: list[dict[str, Any]],
    trace: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    names = [str(r["name"]) for r in active_components(config)]
    sim_cv = np.asarray(trace["CV"], dtype=float)
    comp = np.asarray(trace["component_g_L"], dtype=float)
    rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    for ref in references:
        cv = float(ref["CV"])
        if cv < float(np.min(sim_cv)) - 1e-12 or cv > float(np.max(sim_cv)) + 1e-12:
            raise ValueError(f"Composition reference at CV {cv:g} lies outside the simulated CV range.")
        values = np.array([np.interp(cv, sim_cv, comp[:, i]) for i in range(len(names))], dtype=float)
        total = float(np.sum(values))
        if total <= 1e-12:
            predicted = np.zeros(len(names))
        else:
            predicted = 100.0 * values / total
        for i, name in enumerate(names):
            measured = float(ref["mass_percent"][name])
            pred = float(predicted[i])
            residuals.append((measured - pred) / 100.0)
            rows.append({"CV": cv, "Protein_Species": name, "Measured_mass_percent": measured, "Simulated_mass_percent": pred})
    if residuals:
        arr = np.asarray(residuals, dtype=float) / math.sqrt(len(residuals))
    else:
        arr = np.empty(0, dtype=float)
    return arr, rows


def _objective_parts(
    cfg: dict[str, Any],
    ref: ChromatogramReference,
    mode: str,
    composition_refs: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    result = simulate(cfg)
    chrom_r, chrom_info = _chromatogram_residuals(cfg, ref, result["trace"])
    comp_r = np.empty(0, dtype=float)
    comp_rows: list[dict[str, Any]] = []
    if mode == MODE_CHROMATOGRAM_AND_COMPOSITION:
        comp_r, comp_rows = _composition_residuals(cfg, composition_refs, result["trace"])
    combined = np.concatenate([chrom_r, comp_r])
    return combined, {
        "result": result,
        "chromatogram": chrom_info,
        "composition": comp_rows,
        "chromatogram_sse_normalized": float(np.sum(chrom_r**2)),
        "chromatogram_raw_sse": float(np.sum((chrom_info["predicted_detector_signal"] - chrom_info["measured"])**2)),
        "composition_sse_normalized": float(np.sum(comp_r**2)) if len(comp_r) else 0.0,
    }


def _write_apply_jsl(config: dict[str, Any], specs: list[FitParameter]) -> None:
    mapping = {
        "qmax_g_L": "Qmax", "b_L_g": "B", "Z_ref": "Zref", "delta_ref": "Delta",
        "As_m_inv": "As", "kkin_star_s": "Kkin", "salt_sensitivity_per_M": "SaltSensitivity",
        "uv_response_factor_mAU_L_g": "UVResponse", "D_ax_mm2_s": "Dax",
        "k_eff_um_s": "Keff", "accessible_particle_porosity": "EpsPi",
        "D_app_mm2_s": "Dapp", "diameter_nm": "Diameter", "pH_ref": "PHref",
        "Z1_per_pH": "Z1", "Z2_per_pH2": "Z2", "Z3_per_pH3": "Z3",
        "delta_pH_slope_m2_C": "DeltaSlope",
    }
    global_mapping = {
        "conversion.uv_to_protein_mAU_L_g": "uvToProtein",
        "conversion.conductivity_to_salt_M_per_mS_cm": "condToSalt",
        "tdm.bead_radius_um": "beadRadius",
        "tdm.void_fraction": "voidFraction",
        "tdm.particle_porosity": "particlePorosity",
        "tdm.salt_axial_dispersion_mm2_s": "saltDax",
        "edm.total_bed_porosity": "edmPorosity",
        "cpa.ligand_surface_density_umol_m2": "ligandSurface",
        "cpa.system_specific_adsorption_parameter": "systemAdsorption",
        "system.buffer_dispersion_mL": "bufferDispersion",
        "system.salt_dispersion_mm2_s": "systemSaltDax",
        "system.uv_dead_volume_mL": "uvDeadVolume",
        "system.conductivity_dead_volume_mL": "condDeadVolume",
    }
    lines = ["// Generated by least-squares refinement; updates only fitted mechanistic parameter boxes."]
    for spec in specs:
        tokens = _parse_path(spec.path)
        if tokens[0] == "components":
            comp_index = int(tokens[1]) + 1
            field = str(tokens[-1])
            suffix = mapping[field]
            variable = f"c{comp_index}{suffix}"
        else:
            variable = global_mapping[spec.path]
        value = float(_get_path(config, spec.path))
        lines.append(f"{variable} << Set({value:.17g});")
    lines.append("UpdateUI();")
    FIT_APPLY_JSL.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_output_path(path: Path, run_context: dict[str, Any] | None) -> Path:
    if not run_context or run_context.get("run_number") is None:
        return path
    try:
        run_number = int(run_context["run_number"])
    except (TypeError, ValueError):
        return path
    return path.with_name(f"{path.stem}_run_{run_number:02d}{path.suffix}")


def _objective_history_svg(history: list[dict[str, Any]]) -> str:
    """Create a dependency-free plot of the best objective versus evaluation."""
    width, height = 900, 300
    left, right, top, bottom = 82, 26, 24, 54
    plot_w, plot_h = width - left - right, height - top - bottom
    valid = [row for row in history if math.isfinite(float(row.get("best_objective", math.nan)))]
    if not valid:
        return ""
    xs = [float(row["evaluation"]) for row in valid]
    ys = [max(float(row["best_objective"]), 1e-16) for row in valid]
    xmin, xmax = min(xs), max(xs)
    if xmax <= xmin:
        xmax = xmin + 1.0
    log_values = [math.log10(v) for v in ys]
    ymin, ymax = min(log_values), max(log_values)
    if ymax - ymin < 0.15:
        ymin -= 0.075
        ymax += 0.075
    else:
        pad = 0.06 * (ymax - ymin)
        ymin -= pad
        ymax += pad

    def px(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def py(value: float) -> float:
        return top + (ymax - math.log10(max(value, 1e-16))) / (ymax - ymin) * plot_h

    points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(xs, ys))
    grid = []
    for j in range(4):
        y = top + j * plot_h / 3
        value = 10 ** (ymax - j * (ymax - ymin) / 3)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e1e5e9"/>'
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.3g}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" '
        'aria-label="Best least-squares objective by evaluation, lower is better">'
        f'<rect width="{width}" height="{height}" fill="white"/>{"".join(grid)}'
        f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="2.5"/>'
        f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" font-size="13">Residual evaluation</text>'
        f'<text transform="translate(18 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-size="13">Best objective (log scale)</text>'
        f'<text x="{left}" y="{height-30}" text-anchor="start" font-size="11">{int(xmin)}</text>'
        f'<text x="{left+plot_w}" y="{height-30}" text-anchor="end" font-size="11">{int(max(xs))}</text>'
        '</svg>'
    )


def _objective_improvement_percent(initial: float, candidate: float) -> float:
    """Return a stable percentage change, treating machine-scale objectives as zero."""
    initial = float(initial)
    candidate = float(candidate)
    if abs(initial) <= 1e-12 and abs(candidate - initial) <= 1e-12:
        return 0.0
    return (initial - candidate) / max(abs(initial), 1e-30) * 100.0


def _bounded_jacobian(fun, lower, upper):
    def jacobian(vector):
        base = fun(vector)
        columns = []
        for i, value in enumerate(vector):
            h = max(1e-3, abs(float(value)) * 1e-3)
            trial = np.asarray(vector, dtype=float).copy()
            step = min(h, upper[i] - value)
            if step < h * 0.1:
                step = -min(h, value - lower[i])
            trial[i] += step
            columns.append((fun(trial) - base) / step)
        return np.column_stack(columns)
    return jacobian


def _optimizer_method(initial: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> str:
    """A zero-valued dispersion parameter must be able to leave its bound.

    TRF starts infinitesimally inside an active bound and can make only tiny
    steps for many evaluations. Dogbox takes a finite feasible step from the
    entered value; use TRF for the usual interior starting points.
    """
    return "dogbox" if np.any((initial <= lower) | (initial >= upper)) else "trf"


def run_least_squares_refinement(
    config: dict[str, Any],
    *,
    mode: str,
    chromatogram_csv: Path | str = ATTACHED_CHROMATOGRAM_CSV,
    composition_references: Iterable[dict[str, Any]] | None = None,
    run_context: dict[str, Any] | None = None,
    max_nfev: int = 60,
    unlocked_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    # Canonicalize legacy configs before validation, fitting and writing the
    # parameter receipt. This fills newly added impurity UV factors from the
    # shared target response factor and prevents a None initial value in reports.
    config = normalize_config(config)
    mode = str(mode or "").strip().upper()
    if mode not in SUPPORTED_MODES:
        raise ValueError("Least-squares mode must be CHROMATOGRAM or CHROMATOGRAM_AND_COMPOSITION.")
    errors = validate_config(config, strict=True)
    if errors:
        raise ValueError("Model inputs must be complete before refinement:\n- " + "\n- ".join(errors))
    if mode == MODE_CHROMATOGRAM_AND_COMPOSITION and config["least_squares"]["objective"] == "WEIGHTED_RMSE":
        raise ValueError("WEIGHTED_RMSE is defined for chromatogram signals only. For species mass% constraints select RAW_SSE or NORMALIZED_MSE.")
    fit_results_path = _profile_output_path(FIT_RESULTS_CSV, run_context)
    fit_trace_path = _profile_output_path(FIT_TRACE_CSV, run_context)
    fit_composition_path = _profile_output_path(FIT_COMPOSITION_CSV, run_context)
    fit_report_path = _profile_output_path(FIT_REPORT_HTML, run_context)
    fit_config_path = _profile_output_path(FIT_CONFIG_JSON, run_context)
    fit_settings_path = _profile_output_path(FIT_SETTINGS_JSON, run_context)
    fit_history_path = _profile_output_path(FIT_HISTORY_CSV, run_context)
    fit_apply_profile_path = _profile_output_path(FIT_APPLY_JSL, run_context)
    ref = load_chromatogram_reference(chromatogram_csv, config)
    comp_refs: list[dict[str, Any]] = []
    if mode == MODE_CHROMATOGRAM_AND_COMPOSITION:
        comp_refs = validate_composition_references(config, composition_references or [])

    if unlocked_paths is not None:
        unlocked_paths = list(unlocked_paths)
        config["least_squares"]["unlocked_paths"] = list(unlocked_paths)
    specs = fit_parameter_specs(config, fit_uv_response=not _is_concentration_signal(ref.signal_column), unlocked_paths=unlocked_paths)
    x0, lo, hi = _encode_initial(specs)
    fit_started = time.perf_counter()
    initial_r, initial_info = _objective_parts(config, ref, mode, comp_refs)
    initial_obj = float(np.sum(initial_r**2))
    best_objective = initial_obj
    best_vector = None  # None preserves the original config without a transform round-trip.
    history: list[dict[str, Any]] = [{
        "evaluation": 0, "elapsed_seconds": time.perf_counter() - fit_started,
        "objective": initial_obj, "best_objective": initial_obj,
        "improvement_percent": 0.0, "phase": "initial", "status": "ok",
    }]
    eval_count = 0

    def record_objective(objective: float, phase: str, status: str) -> None:
        nonlocal best_objective
        best_objective = min(best_objective, objective)
        improvement = _objective_improvement_percent(initial_obj, best_objective)
        history.append({
            "evaluation": eval_count,
            "elapsed_seconds": time.perf_counter() - fit_started,
            "objective": objective,
            "best_objective": best_objective,
            "improvement_percent": improvement,
            "phase": phase,
            "status": status,
        })

    def residual(vector: np.ndarray) -> np.ndarray:
        nonlocal eval_count, best_vector
        eval_count += 1
        try:
            candidate = _apply_vector(config, specs, vector)
            r, info = _objective_parts(candidate, ref, mode, comp_refs)
            objective = float(np.sum(r*r))
            if not np.all(np.isfinite(r)):
                raise FloatingPointError("non-finite residual")
            if not math.isfinite(objective):
                raise FloatingPointError("non-finite objective")
            if objective < best_objective:
                best_vector = np.asarray(vector).copy()
            record_objective(objective, "optimizer", "ok")
            return r
        except Exception as exc:
            # Keep the optimizer within the physically/numerically feasible region.
            penalty = np.full_like(initial_r, 1e6, dtype=float)
            record_objective(float(np.sum(penalty * penalty)), "optimizer", f"penalty: {type(exc).__name__}: {exc}")
            return penalty

    if specs:
        opt = least_squares(
            residual,
            x0,
            bounds=(lo, hi),
            jac=_bounded_jacobian(residual, lo, hi),
            method=_optimizer_method(x0, lo, hi),
            loss="linear",
            x_scale="jac",
            # The model is integrated numerically at finite tolerance.  SciPy's
            # default machine-epsilon step can be smaller than the ODE noise and
            # produce a zero/unstable Jacobian, causing premature `xtol` stops.
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            max_nfev=max(2, int(max_nfev)),
        )
    else:
        opt = SimpleNamespace(x=x0, success=True, message="All eligible parameters are locked; evaluated without refinement.", nfev=0)

    if best_vector is None:
        fitted = copy.deepcopy(config)
        opt.x = np.asarray([math.log10(float(_get_path(config, spec.path))) if spec.transform == "LOG10"
                            else float(_get_path(config, spec.path)) for spec in specs])
    else:
        opt.x = best_vector.copy()
        fitted = _apply_vector(config, specs, opt.x)
    final_r, final_info = _objective_parts(fitted, ref, mode, comp_refs)
    final_obj = float(np.sum(final_r**2))
    eval_count += 1
    record_objective(final_obj, "final verification", "ok")

    # Never replace the active model with a numerically worse refinement.
    # scipy may stop at max_nfev with success=False while still having found an
    # improved parameter set, so acceptance is based on the objective itself.
    accepted = bool(math.isfinite(final_obj) and final_obj <= initial_obj + 32.0 * np.finfo(float).eps * max(initial_obj, np.finfo(float).tiny))
    applied = fitted if accepted else copy.deepcopy(config)

    # Keep the optimizer result as an audit receipt, but only make it the active
    # model (and update the JMP boxes) when it did not worsen the objective.
    save_config(fitted, fit_config_path)
    if accepted:
        save_config(fitted, CONFIG_PATH)
        _write_apply_jsl(fitted, specs)
    elif FIT_APPLY_JSL.exists():
        FIT_APPLY_JSL.unlink()
    if not accepted and fit_apply_profile_path.exists():
        fit_apply_profile_path.unlink()

    with fit_results_path.open("w", encoding="utf-8", newline="") as fh:
        fields = ["Parameter_Path", "Parameter", "Initial_Value", "Fitted_Value", "Transform", "At_Lower_Bound", "At_Upper_Bound"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for spec, xfit in zip(specs, opt.x):
            initial_value = float(_get_path(config, spec.path))
            fitted_value = float(_get_path(fitted, spec.path))
            writer.writerow({
                "Parameter_Path": spec.path,
                "Parameter": spec.label,
                "Initial_Value": initial_value,
                "Fitted_Value": fitted_value,
                "Transform": spec.transform,
                "At_Lower_Bound": abs(float(xfit) - spec.lower) <= 1e-6 * max(1.0, abs(spec.lower)),
                "At_Upper_Bound": abs(float(xfit) - spec.upper) <= 1e-6 * max(1.0, abs(spec.upper)),
            })

    chrom = final_info["chromatogram"]
    with fit_trace_path.open("w", encoding="utf-8", newline="") as fh:
        fields = ["CV", "Measured_Signal", "Simulated_Total_Protein_g_L", "Scaled_Simulated_Signal", "Residual"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for cv, ym, ys, yp in zip(chrom["CV"], chrom["measured"], chrom["simulated_total_g_L"], chrom["predicted_detector_signal"]):
            writer.writerow({"CV": cv, "Measured_Signal": ym, "Simulated_Total_Protein_g_L": ys, "Scaled_Simulated_Signal": yp, "Residual": float(yp)-float(ym)})

    with fit_composition_path.open("w", encoding="utf-8", newline="") as fh:
        fields = ["CV", "Protein_Species", "Measured_mass_percent", "Simulated_mass_percent", "Residual_mass_percent"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for row in final_info["composition"]:
            r = dict(row)
            r["Residual_mass_percent"] = float(r["Measured_mass_percent"]) - float(r["Simulated_mass_percent"])
            writer.writerow(r)

    with fit_history_path.open("w", encoding="utf-8", newline="") as fh:
        history_fields = ["evaluation", "elapsed_seconds", "objective", "best_objective", "improvement_percent", "phase", "status"]
        writer = csv.DictWriter(fh, fieldnames=history_fields)
        writer.writeheader()
        writer.writerows(history)

    settings = {
        "mode": mode,
        "objective": config["least_squares"]["objective"],
        "baseline_mode": config["least_squares"]["baseline_mode"],
        "fitted_parameter_paths": [s.path for s in specs],
        "equation_9": "sum_points (predicted - measured)^2; only unlocked eligible parameters vary",
        "reference_csv": str(Path(chromatogram_csv).resolve()),
        "reference_x_column": ref.x_column,
        "reference_signal_column": ref.signal_column,
        "reference_x_unit": ref.x_unit,
        "reference_x_axis_assumption": ref.x_axis_assumption,
        "composition_references": comp_refs,
        "fitted_parameter_paths": [s.path for s in specs],
        "max_nfev": int(max_nfev),
        "objective_history_csv": str(fit_history_path),
        "target_run_context": run_context,
    }
    fit_settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    parameter_rows = "".join(
        f"<tr><td>{html.escape(spec.label)}</td><td>{float(_get_path(config, spec.path)):.7g}</td><td>{float(_get_path(fitted, spec.path)):.7g}</td></tr>"
        for spec in specs
    )
    comp_note = (
        f"<p><b>Composition constraints:</b> {len(comp_refs)} CV points, {len(final_info['composition'])} species observations. "
        f"Normalised composition SSE = {final_info['composition_sse_normalized']:.6g}.</p>"
        if mode == MODE_CHROMATOGRAM_AND_COMPOSITION else
        "<p><b>Composition constraints:</b> not used in this refinement.</p>"
    )
    x_assumption_note = (
        f"<p><b>X-axis assumption:</b> {html.escape(ref.x_axis_assumption)}</p>"
        if ref.x_axis_assumption else ""
    )
    run_context_html = (
        "<h2>Fit target run and operating recipe</h2>"
        "<p>The selected run's column, feed, numerical settings, buffers, load chemistry, "
        "PLW steps, and elution steps were held fixed while adsorption parameters were fitted.</p>"
        f"<details open><summary>Run recipe snapshot</summary><pre>{html.escape(json.dumps(run_context, indent=2, ensure_ascii=False))}</pre></details>"
        if run_context is not None else
        "<p><b>Fit target run:</b> not supplied by direct Python call.</p>"
    )
    improvement_percent = _objective_improvement_percent(initial_obj, final_obj)
    history_plot = _objective_history_svg(history)
    uv_identifiability_note = (
        "<p>Impurity UV factors are estimated from the composite detector trace. When peaks overlap, their factors can be correlated with each other and with adsorption parameters; species-resolved UV data or composition constraints improve identifiability.</p>"
        if any(spec.path.endswith("uv_response_factor_mAU_L_g") for spec in specs) else
        "<p>Impurity UV factors were held fixed by the lock selection or because the attached signal column is concentration-valued. Only eligible unlocked response factors are refined.</p>"
    )
    fit_report_path.write_text(f"""<!doctype html>
<meta charset="utf-8"><title>Least-squares refinement</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;max-width:1100px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.35rem .55rem;text-align:right}}td:first-child,th:first-child{{text-align:left}}code,pre{{background:#f4f4f4;padding:.2rem .35rem}}</style>
<h1>Least-squares refinement</h1>
<p><b>Mode:</b> {html.escape(mode)}</p>
<p><b>Reference chromatogram:</b> {html.escape(str(Path(chromatogram_csv).resolve()))}<br>
Detected x column: <code>{html.escape(ref.x_column)}</code> ({html.escape(ref.x_unit)}); signal column: <code>{html.escape(ref.signal_column)}</code>.</p>
{x_assumption_note}
{run_context_html}
<p><b>Optimiser:</b> SciPy bounded nonlinear least_squares, linear loss.<br>
Converged flag: {bool(opt.success)}; accepted/applied to active model: {accepted}; function evaluations reported by optimiser: {int(opt.nfev)}; objective evaluations including final check: {eval_count}.</p>
<p><b>Initial total objective:</b> {initial_obj:.7g}<br><b>Final total objective:</b> {final_obj:.7g}<br>
<b>Improvement:</b> {improvement_percent:.4g}%<br><b>Final chromatogram objective ({html.escape(config['least_squares']['objective'])}):</b> {final_info['chromatogram_sse_normalized']:.7g}</p>
{comp_note}
<h2>Refined mechanistic parameters</h2>
<table><thead><tr><th>Parameter</th><th>Initial</th><th>Fitted</th></tr></thead><tbody>{parameter_rows}</tbody></table>
<h2>Objective construction</h2>
<p>The chromatogram residual is point-by-point least squares (Hahn et al., equation 9). RAW_SSE is the default. No baseline is inferred unless INITIAL_MEDIAN is explicitly selected; otherwise the fixed detector baseline is used. For UV data, the predicted detector signal is the sum of each species concentration multiplied by its own response factor; the target uses the shared Model Parameters factor and each impurity uses its component-specific factor. Concentration-valued CSV columns are compared directly. Optional NORMALIZED_MSE divides residuals by the measured signal scale and sqrt(N). RAW_SSE compares detector values directly.</p>
{uv_identifiability_note}
<p>When composition constraints are selected, simulated species mass% is evaluated at each entered CV as 100 × c_i / sum(c_j). Those residuals are divided by 100 and sqrt(Ncomposition), giving the composition block equal group-level footing with the chromatogram block.</p>
<p>Only unlocked eligible mechanistic and instrument parameters listed above are refined. Locked values, column geometry and each recipe remain fixed. The optimizer result was saved to <code>{html.escape(fit_config_path.name)}</code>. It is copied to the active model <code>{html.escape(CONFIG_PATH.name)}</code> only when the total objective is not worse than the starting objective.</p>
<h2>Objective improvement over time</h2>
<p>The plot and CSV show every objective evaluation, elapsed seconds, best-so-far objective, improvement relative to the initial objective, and any numerical-penalty status. Lower objective is better. <a href="{html.escape(fit_history_path.name)}">Open objective history CSV</a>.</p>
{history_plot}
<p>Files: <code>{fit_trace_path.name}</code>, <code>{fit_composition_path.name}</code>, <code>{fit_results_path.name}</code>, <code>{fit_history_path.name}</code>.</p>
""", encoding="utf-8")

    # Keep the existing "latest fit" files for the UI buttons while retaining
    # one permanent set per saved chromatogram profile.
    if run_context and fit_report_path != FIT_REPORT_HTML:
        for source_path, latest_path in (
            (fit_results_path, FIT_RESULTS_CSV),
            (fit_trace_path, FIT_TRACE_CSV),
            (fit_composition_path, FIT_COMPOSITION_CSV),
            (fit_report_path, FIT_REPORT_HTML),
            (fit_config_path, FIT_CONFIG_JSON),
            (fit_settings_path, FIT_SETTINGS_JSON),
            (fit_history_path, FIT_HISTORY_CSV),
        ):
            if source_path.is_file():
                shutil.copyfile(source_path, latest_path)
        if accepted and FIT_APPLY_JSL.is_file():
            shutil.copyfile(FIT_APPLY_JSL, fit_apply_profile_path)

    return {
        "success": bool(opt.success),
        "accepted": accepted,
        "message": str(opt.message),
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "improvement_percent": improvement_percent,
        "chromatogram_sse": float(final_info["chromatogram_sse_normalized"]),
        "composition_sse": float(final_info["composition_sse_normalized"]),
        "nfev": int(opt.nfev),
        "residual_evaluations": eval_count,
        "mode": mode,
        "target_run_context": run_context,
        "fitted_config": fitted,
        "fit_results_csv": str(fit_results_path),
        "fit_trace_csv": str(fit_trace_path),
        "fit_composition_csv": str(fit_composition_path),
        "fit_report_html": str(fit_report_path),
        "fit_config_json": str(fit_config_path),
        "fit_settings_json": str(fit_settings_path),
        "fit_history_csv": str(fit_history_path),
        "latest_fit_history_csv": str(FIT_HISTORY_CSV),
        "apply_jsl": str(FIT_APPLY_JSL),
        "profile_apply_jsl": str(fit_apply_profile_path),
        "parameters": [s.path for s in specs],
    }


def run_global_least_squares_refinement(
    profiles: Iterable[dict[str, Any]],
    *,
    mode: str = MODE_CHROMATOGRAM,
    unlocked_paths: Iterable[str] | None = None,
    max_nfev: int = 60,
) -> dict[str, Any]:
    """Fit one shared mechanistic parameter vector against 1–5 run profiles.

    Each profile carries its own fixed recipe and CSV/Excel workbook. By default the point
    residuals are concatenated without scaling: their squared norm is equation
    9 summed across runs. NORMALIZED_MSE is an explicit optional weighting.
    """
    profiles = list(profiles)
    if not 1 <= len(profiles) <= 5:
        raise ValueError("Global least-squares fitting requires 1–5 attached chromatogram profiles.")
    mode = str(mode or "").strip().upper()
    if mode not in SUPPORTED_MODES:
        raise ValueError("Least-squares mode must be CHROMATOGRAM or CHROMATOGRAM_AND_COMPOSITION.")

    normalized_profiles: list[dict[str, Any]] = []
    candidate_paths: list[str] = []
    refs: list[ChromatogramReference] = []
    for index, profile in enumerate(profiles, start=1):
        cfg = normalize_config(profile.get("config"))
        if mode == MODE_CHROMATOGRAM_AND_COMPOSITION and cfg["least_squares"]["objective"] == "WEIGHTED_RMSE":
            raise ValueError("WEIGHTED_RMSE is defined for chromatogram signals only. Select RAW_SSE or NORMALIZED_MSE to include species mass% constraints.")
        errors = validate_config(cfg, strict=True)
        if errors:
            context = profile.get("run_context") or {}
            label = context.get("run_name") or f"Profile {index}"
            raise ValueError(f"Global-fit profile {label} has incomplete model inputs:\n- " + "\n- ".join(errors))
        source = profile.get("chromatogram_csv") or profile.get("csv_path")
        if not source:
            raise ValueError(f"Global-fit profile {index} has no chromatogram CSV path.")
        ref = load_chromatogram_reference(source, cfg,
                                         sheet_name=profile.get("sheet_name"),
                                         x_column=profile.get("x_column"),
                                         signal_column=profile.get("signal_column"),
                                         x_unit=profile.get("x_unit"),
                                         x_origin=profile.get("x_origin"))
        profile_weight = float(profile.get("weight", 1.0))
        if not math.isfinite(profile_weight) or profile_weight <= 0:
            raise ValueError(f"Global-fit profile {index}: run weight must be finite and > 0.")
        raw_composition_refs = profile.get("composition_references", [])
        profile_refs = validate_composition_references(cfg, raw_composition_refs) if mode == MODE_CHROMATOGRAM_AND_COMPOSITION and raw_composition_refs else []
        context = dict(profile.get("run_context") or {})
        context.setdefault("run_number", index)
        context.setdefault("run_name", f"Profile {index}")
        context["chromatogram_csv"] = str(Path(source).resolve())
        paths = [path for path in required_parameter_paths(cfg)
                 if path not in {"column.volume_mL", "column.length_mm"} and not path.endswith(".mass_percent")]
        for path in paths:
            if path not in candidate_paths:
                candidate_paths.append(path)
        normalized_profiles.append({
            "config": cfg,
            "reference": ref,
            "composition_references": profile_refs,
            "run_context": context,
            "weight": profile_weight,
            "x_origin": _validate_x_origin(profile.get("x_origin"), ref.x_unit),
        })

    if mode == MODE_CHROMATOGRAM_AND_COMPOSITION and not any(p["composition_references"] for p in normalized_profiles):
        raise ValueError("Supply at least one composition reference for a selected profile when chromatogram + mass% is selected.")
    first = normalized_profiles[0]["config"]
    if mode == MODE_CHROMATOGRAM_AND_COMPOSITION and first["least_squares"]["objective"] == "WEIGHTED_RMSE":
        raise ValueError("WEIGHTED_RMSE is defined for chromatogram signals only. For additional species mass% constraints select RAW_SSE or NORMALIZED_MSE, because composition residuals have different units.")
    for index, profile in enumerate(normalized_profiles[1:], start=2):
        cfg = profile["config"]
        if cfg["model"] != first["model"] or int(cfg.get("impurity_count", 0)) != int(first.get("impurity_count", 0)):
            raise ValueError("All attached chromatograms must use the same model combination and selected species for one global shared fit.")
        if [r.get("name") for r in active_components(cfg)] != [r.get("name") for r in active_components(first)]:
            raise ValueError(f"Global-fit profile {index} does not have the same shared protein-species names as profile 1.")

    # These are shared batch parameters. Process recipes and column geometry
    # remain profile inputs, while a different non-empty value for a shared
    # mechanistic parameter is rejected instead of silently choosing one run.
    seed = copy.deepcopy(first)
    for path in candidate_paths:
        values = []
        for profile in normalized_profiles:
            try:
                value = _get_path(profile["config"], path)
            except (KeyError, IndexError, TypeError):
                value = None
            if value is not None:
                values.append(value)
        if not values:
            continue
        reference_value = values[0]
        for value in values[1:]:
            try:
                same = math.isclose(float(reference_value), float(value), rel_tol=1e-10, abs_tol=1e-12)
            except (TypeError, ValueError):
                same = reference_value == value
            if not same:
                raise ValueError(f"Shared mechanistic parameter {path} differs between attached profiles. Global fitting requires one common starting value.")
        _set_path(seed, path, reference_value)
        for profile in normalized_profiles:
            _set_path(profile["config"], path, reference_value)

    if unlocked_paths is not None:
        unlocked_paths = list(unlocked_paths)
        for profile in normalized_profiles:
            profile["config"]["least_squares"]["unlocked_paths"] = list(unlocked_paths)
        seed["least_squares"]["unlocked_paths"] = list(unlocked_paths)
    has_uv = any(not _is_concentration_signal(p["reference"].signal_column) for p in normalized_profiles)
    specs = fit_parameter_specs(
        seed,
        fit_uv_response=has_uv,
        unlocked_paths=unlocked_paths,
        candidate_paths=candidate_paths,
    )
    x0, lo, hi = _encode_initial(specs)
    n_profiles = len(normalized_profiles)
    normalized_objective = first["least_squares"]["objective"] == "NORMALIZED_MSE"
    weighted_objective = first["least_squares"]["objective"] == "WEIGHTED_RMSE"
    run_weights = np.array([p["weight"] for p in normalized_profiles], dtype=float)
    run_weights /= float(np.sum(run_weights))
    if weighted_objective and len({_is_concentration_signal(p["reference"].signal_column) for p in normalized_profiles}) != 1:
        raise ValueError("Weighted RMSE cannot combine concentration-valued and UV detector chromatograms: their signal units differ.")
    for profile in normalized_profiles:
        if profile["config"]["least_squares"]["objective"] != first["least_squares"]["objective"] or profile["config"]["chromatography_mode"] != first["chromatography_mode"]:
            raise ValueError("Global profiles must use the same objective and chromatography mode.")
    comp_refs_by_profile: list[list[dict[str, Any]]] = [p["composition_references"] for p in normalized_profiles]

    def evaluate(configs: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        residual_blocks = []
        infos = []
        for cfg, prof, comp_refs, normalized_weight in zip(configs, normalized_profiles, comp_refs_by_profile, run_weights):
            r, info = _objective_parts(cfg, prof["reference"], mode, comp_refs)
            # WEIGHTED_RMSE: sum_r alpha_r * (sum_j w_rj e_rj² / sum_j w_rj).
            # Minimizing its square root is equivalent to minimizing its SSE
            # residual vector (SciPy's least_squares contract).
            scale_factor = math.sqrt(float(normalized_weight)) if weighted_objective else (1.0 / math.sqrt(n_profiles) if normalized_objective else 1.0)
            residual_blocks.append(r * scale_factor)
            infos.append(info)
        return np.concatenate(residual_blocks), infos

    base_configs = [copy.deepcopy(p["config"]) for p in normalized_profiles]
    initial_r, initial_infos = evaluate(base_configs)
    initial_obj = float(np.sum(initial_r ** 2))
    fit_started = time.perf_counter()
    best_objective = initial_obj
    best_vector = None  # No improved candidate: retain exact original numeric values.
    history: list[dict[str, Any]] = [{
        "evaluation": 0, "elapsed_seconds": 0.0, "objective": initial_obj,
        "best_objective": initial_obj, "improvement_percent": 0.0,
        "phase": "initial", "status": "ok",
    }]
    eval_count = 0

    def write_history() -> None:
        FIT_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        with FIT_HISTORY_CSV.open("w", encoding="utf-8", newline="") as fh:
            fields = ["evaluation", "elapsed_seconds", "objective", "best_objective", "improvement_percent", "phase", "status"]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)

    def record(objective: float, phase: str, status: str) -> None:
        nonlocal best_objective
        best_objective = min(best_objective, objective)
        history.append({
            "evaluation": eval_count,
            "elapsed_seconds": time.perf_counter() - fit_started,
            "objective": objective,
            "best_objective": best_objective,
            "improvement_percent": _objective_improvement_percent(initial_obj, best_objective),
            "phase": phase,
            "status": status,
        })
        write_history()

    write_history()

    def residual(vector: np.ndarray) -> np.ndarray:
        nonlocal eval_count, best_vector
        eval_count += 1
        try:
            candidates = [_apply_vector(cfg, specs, vector) for cfg in base_configs]
            values, _infos = evaluate(candidates)
            objective = float(np.sum(values * values))
            if not np.all(np.isfinite(values)) or not math.isfinite(objective):
                raise FloatingPointError("non-finite global residual")
            if objective < best_objective:
                best_vector = np.asarray(vector).copy()
            record(objective, "optimizer", "ok")
            return values
        except Exception as exc:
            penalty = np.full_like(initial_r, 1e6, dtype=float)
            record(float(np.sum(penalty * penalty)), "optimizer", f"penalty: {type(exc).__name__}: {exc}")
            return penalty

    if specs:
        opt = least_squares(
            residual,
            x0,
            bounds=(lo, hi),
            jac=_bounded_jacobian(residual, lo, hi),
            method=_optimizer_method(x0, lo, hi),
            loss="linear",
            x_scale="jac",
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            max_nfev=max(2, int(max_nfev)),
        )
        if best_vector is None:
            fitted_configs = [copy.deepcopy(cfg) for cfg in base_configs]
            opt.x = np.asarray([math.log10(float(_get_path(seed, spec.path))) if spec.transform == "LOG10"
                                else float(_get_path(seed, spec.path)) for spec in specs])
        else:
            opt.x = best_vector.copy()
            fitted_configs = [_apply_vector(cfg, specs, opt.x) for cfg in base_configs]
        opt_success = bool(opt.success)
        opt_message = str(opt.message)
        opt_nfev = int(opt.nfev)
    else:
        fitted_configs = [copy.deepcopy(cfg) for cfg in base_configs]
        opt_success = True
        opt_message = "All eligible mechanistic parameters are locked; objective evaluated without optimization."
        opt_nfev = 0
    try:
        final_r, final_infos = evaluate(fitted_configs)
        final_obj = float(np.sum(final_r ** 2))
        if not math.isfinite(final_obj):
            raise FloatingPointError("non-finite final global objective")
    except Exception as exc:
        fitted_configs = [copy.deepcopy(cfg) for cfg in base_configs]
        final_r, final_infos = initial_r, initial_infos
        final_obj = initial_obj
        opt_success = False
        opt_message = f"Final candidate verification failed; original parameters retained: {type(exc).__name__}: {exc}"
    eval_count += 1
    record(final_obj, "final verification", "ok" if opt_success else opt_message)
    accepted = bool(math.isfinite(final_obj) and final_obj <= initial_obj + 32.0 * np.finfo(float).eps * max(initial_obj, np.finfo(float).tiny))
    fitted = fitted_configs[0] if accepted else copy.deepcopy(base_configs[0])
    if not accepted:
        fitted_configs = [copy.deepcopy(cfg) for cfg in base_configs]

    save_config(fitted, FIT_CONFIG_JSON)
    if accepted:
        save_config(fitted, CONFIG_PATH)
        _write_apply_jsl(fitted, specs)
    elif FIT_APPLY_JSL.exists():
        FIT_APPLY_JSL.unlink()

    with FIT_RESULTS_CSV.open("w", encoding="utf-8", newline="") as fh:
        fields = ["Parameter_Path", "Parameter", "Initial_Value", "Fitted_Value", "Transform", "Fit_State", "At_Lower_Bound", "At_Upper_Bound"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        spec_map = {spec.path: spec for spec in specs}
        for path in candidate_paths:
            try:
                initial_value = float(_get_path(seed, path))
                fitted_value = float(_get_path(fitted, path))
            except (KeyError, TypeError, ValueError):
                continue
            spec = spec_map.get(path)
            fit_state = "Unlocked" if spec else "Locked"
            xfit = None
            if spec and specs:
                xfit = (10.0 ** float(opt.x[specs.index(spec)]) if spec.transform == "LOG10"
                        else float(opt.x[specs.index(spec)]))
            writer.writerow({
                "Parameter_Path": path, "Parameter": spec.label if spec else path,
                "Initial_Value": initial_value, "Fitted_Value": fitted_value,
                "Transform": spec.transform if spec else "", "Fit_State": fit_state,
                "At_Lower_Bound": bool(spec and xfit is not None and abs(float(opt.x[specs.index(spec)]) - spec.lower) <= 1e-6 * max(1.0, abs(spec.lower))),
                "At_Upper_Bound": bool(spec and xfit is not None and abs(float(opt.x[specs.index(spec)]) - spec.upper) <= 1e-6 * max(1.0, abs(spec.upper))),
            })

    # Only unlocked parameters are listed as improvements; report the exact
    # values in original physical units, even when the fit variable was log10.
    with FIT_IMPROVED_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Parameter_Path", "Parameter", "Initial_Value", "Improved_Value", "Change_percent", "At_Lower_Bound", "At_Upper_Bound"])
        writer.writeheader()
        text_rows = ["LEAST-SQUARES REFINED PARAMETERS", f"Objective: {first['least_squares']['objective']}", f"Accepted: {accepted}", f"Initial objective: {initial_obj:.10g}", f"Final objective: {final_obj:.10g}", ""]
        for i, spec in enumerate(specs):
            initial_val = float(_get_path(seed, spec.path))
            fitted_val = float(_get_path(fitted, spec.path))
            diff_percent = 100.0 * (fitted_val - initial_val) / abs(initial_val) if initial_val != 0 else math.nan
            xfit_val = math.log10(fitted_val) if spec.transform == "LOG10" else fitted_val
            row = {"Parameter_Path": spec.path, "Parameter": spec.label,
                   "Initial_Value": initial_val, "Improved_Value": fitted_val,
                   "Change_percent": diff_percent,
                   "At_Lower_Bound": abs(xfit_val - spec.lower) <= 1e-6 * max(1.0, abs(spec.lower)),
                   "At_Upper_Bound": abs(xfit_val - spec.upper) <= 1e-6 * max(1.0, abs(spec.upper))}
            writer.writerow(row)
            text_rows.append(f"{spec.path} ({spec.label}): {initial_val:.8g} -> {fitted_val:.8g} ({diff_percent:+.5g}%)")
        if not specs:
            text_rows.append("No parameters unlocked for refinement.")
        FIT_IMPROVED_TXT.write_text("\n".join(text_rows) + "\n", encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    with FIT_TRACE_CSV.open("w", encoding="utf-8", newline="") as fh:
        trace_fields = ["Run_Number", "Run_Name", "Chromatogram_CSV", "CV", "Measured_Signal",
                        "Simulated_Total_Protein_g_L", "Predicted_Detector_Signal", "Residual",
                        "Normalized_Residual", "Sample_Weight", "Run_Weight", "Profile_Objective_Contribution"]
        writer = csv.DictWriter(fh, fieldnames=trace_fields)
        writer.writeheader()
        for profile, info, norm_run_weight in zip(normalized_profiles, final_infos, run_weights):
            chrom = info["chromatogram"]
            context = profile["run_context"]
            difference = np.asarray(chrom["predicted_detector_signal"], dtype=float) - np.asarray(chrom["measured"], dtype=float)
            norm_difference = difference / float(chrom["measured_signal_scale"])
            normalized_nrmse = float(np.sqrt(np.mean(difference ** 2))) / float(chrom["measured_signal_scale"])
            raw_rmse = float(np.sqrt(np.mean(difference ** 2)))
            sample_weights = np.asarray(chrom["point_weights"], dtype=float)
            run_weighted_rmse = float(np.sqrt(np.sum(sample_weights * difference ** 2) / np.sum(sample_weights)))
            total_ss = float(np.sum((np.asarray(chrom["measured"], dtype=float) - float(np.mean(chrom["measured"]))) ** 2))
            r2 = 1.0 - float(np.sum(difference ** 2)) / total_ss if total_ss > 1e-30 else math.nan
            metric = {
                "run_number": int(context["run_number"]),
                "run_name": str(context["run_name"]),
                "chromatogram_csv": context["chromatogram_csv"],
                "overlap_points": int(chrom["overlap_points"]),
                "normalized_nrmse": normalized_nrmse,
                "raw_rmse": raw_rmse,
                "weighted_rmse": run_weighted_rmse,
                "run_weight": profile["weight"],
                "r_squared": r2,
                "profile_objective": float(info["chromatogram_sse_normalized"] + info["composition_sse_normalized"]),
            }
            metric_rows.append(metric)
            for cv, measured, sim_total, predicted, residual_value, norm_value, sample_weight in zip(
                chrom["CV"], chrom["measured"], chrom["simulated_total_g_L"],
                chrom["predicted_detector_signal"], difference, norm_difference, sample_weights,
            ):
                writer.writerow({
                    "Run_Number": metric["run_number"], "Run_Name": metric["run_name"],
                    "Chromatogram_CSV": metric["chromatogram_csv"], "CV": float(cv),
                    "Measured_Signal": float(measured), "Simulated_Total_Protein_g_L": float(sim_total),
                    "Predicted_Detector_Signal": float(predicted), "Residual": float(residual_value),
                    "Normalized_Residual": float(norm_value),
                    "Sample_Weight": float(sample_weight), "Run_Weight": float(profile["weight"]),
                    "Profile_Objective_Contribution": (float(norm_run_weight * sample_weight / np.sum(sample_weights) * residual_value ** 2)
                         if weighted_objective else float(norm_value ** 2 / len(chrom["CV"]) / n_profiles) if normalized_objective else float(residual_value ** 2)),
                })

    composition_rows = []
    with FIT_COMPOSITION_CSV.open("w", encoding="utf-8", newline="") as fh:
        fields = ["Run_Number", "Run_Name", "CV", "Protein_Species", "Measured_mass_percent", "Simulated_mass_percent", "Residual_mass_percent"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for profile, info in zip(normalized_profiles, final_infos):
            context = profile["run_context"]
            for raw in info["composition"]:
                row = {
                    "Run_Number": context["run_number"], "Run_Name": context["run_name"],
                    **raw,
                    "Residual_mass_percent": float(raw["Measured_mass_percent"]) - float(raw["Simulated_mass_percent"]),
                }
                composition_rows.append(row)
                writer.writerow(row)

    write_history()
    settings = {
        "mode": mode,
        "profile_count": n_profiles,
        "objective": first["least_squares"]["objective"],
        "equation_9": "min_theta sum_runs sum_points (predicted_signal(theta_fixed, theta_unlocked) - measured_signal)^2",
        "weighted_rmse_formula": "sqrt(sum_runs (run_weight/sum_run_weights) * sum_points(point_weight*residual^2)/sum_point_weights)",
        "optimizer": "scipy.optimize.least_squares(method=trf, loss=linear), bounded residual-vector least squares as in MATLAB lsqnonlin; distinct numerical implementation",
        "baseline_mode": first["least_squares"]["baseline_mode"],
        "profile_weighting": "User run weights times within-run weighted MSE; sqrt gives weighted RMSE" if weighted_objective else "Equal normalized profile MSE" if normalized_objective else "Unweighted raw point SSE (equation 9); all selected measured points contribute",
        "profiles": [
            {
                "run_context": p["run_context"],
                "x_column": p["reference"].x_column,
                "signal_column": p["reference"].signal_column,
                "x_unit": p["reference"].x_unit,
                "x_origin": p["x_origin"],
                "x_axis_assumption": p["reference"].x_axis_assumption,
                "selected_sheet": p.get("sheet_name", "auto"),
                "composition_references": p["composition_references"],
                "run_weight": p["weight"], "point_weight_column": p["reference"].weight_column,
                "recipe": p["config"],
                "baseline_and_objective": p["config"]["least_squares"],
            }
            for p in normalized_profiles
        ],
        "fitted_parameter_paths": [s.path for s in specs],
        "locked_parameter_paths": [p for p in candidate_paths if p not in {s.path for s in specs}],
        "max_nfev": int(max_nfev),
        "objective_history_csv": str(FIT_HISTORY_CSV),
    }
    FIT_SETTINGS_JSON.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    parameter_rows = "".join(
        f"<tr><td>{html.escape(spec.label)}</td><td>{float(_get_path(seed, spec.path)):.7g}</td><td>{float(_get_path(fitted, spec.path)):.7g}</td></tr>"
        for spec in specs
    ) or '<tr><td colspan="3">No parameters were unlocked; model values were evaluated without optimization.</td></tr>'
    profile_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            metric["run_number"], metric["run_name"], metric["overlap_points"],
            f'{metric["normalized_nrmse"]:.6g}', f'{metric["raw_rmse"]:.6g}', f'{metric["weighted_rmse"]:.6g}', f'{metric["run_weight"]:.6g}',
            "—" if not math.isfinite(metric["r_squared"]) else f'{metric["r_squared"]:.6g}',
        )) + "</tr>"
        for metric in metric_rows
    )
    recipe_details = "".join(
        f'<details><summary>Run {p["run_context"]["run_number"]} — {html.escape(p["run_context"]["run_name"])} recipe and input CSV</summary>'
        f'<pre>{html.escape(json.dumps({"run_context": p["run_context"], "configuration": p["config"]}, indent=2, ensure_ascii=False))}</pre></details>'
        for p in normalized_profiles
    )
    profile_plots = []
    for profile, cfg, info in zip(normalized_profiles, fitted_configs, final_infos):
        chrom = info["chromatogram"]
        trace = info["result"]["trace"]
        program = info["result"]["simulation"]["program"]
        x = chrom["CV"]
        plot = _svg_plot(x, [("Measured signal", chrom["measured"]), ("Predicted signal", chrom["predicted_detector_signal"])],
                         stages=_stage_receipt(cfg, program), total_cv=program.total_CV,
                         percent_B=np.interp(x, trace["CV"], trace["programmed_percent_B"]),
                         conductivity=np.interp(x, trace["CV"], trace["conductivity_mS_cm"]),
                         time_min=np.interp(x, trace["CV"], trace["time_min"]),
                         y_axis_label="Protein concentration (g/L)" if _is_concentration_signal(profile["reference"].signal_column) else "UV absorbance (mAU)")
        profile_plots.append(f'<h3>Run {profile["run_context"]["run_number"]}: {html.escape(profile["run_context"]["run_name"])}</h3>' + plot)
    overlays_html = "".join(profile_plots)
    improvement = _objective_improvement_percent(initial_obj, final_obj)
    history_plot = _objective_history_svg(history)
    FIT_REPORT_HTML.write_text(f"""<!doctype html>
<meta charset="utf-8"><title>Global least-squares refinement</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;max-width:1100px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.35rem .55rem;text-align:right}}td:first-child,th:first-child{{text-align:left}}code,pre{{background:#f4f4f4;padding:.2rem .35rem}}details{{margin:.6rem 0}}</style>
<h1>Global least-squares refinement</h1>
<p><b>Mode:</b> {html.escape(mode)}; <b>Profiles:</b> {n_profiles}; <b>Model:</b> {html.escape(first["model"])}</p>
<p><b>Initial global objective:</b> {initial_obj:.7g}<br><b>Final global objective:</b> {final_obj:.7g}<br><b>Improvement:</b> {improvement:.4g}%<br>
<b>Accepted/applied:</b> {accepted}; <b>Optimizer converged:</b> {opt_success}; {html.escape(opt_message)}<br><b>Optimizer evaluations:</b> {eval_count}; <b>Unlocked parameters:</b> {len(specs)}.</p>
<h2>Per-profile chromatogram metrics</h2><p>Objective: {html.escape(first["least_squares"]["objective"])}. Baseline policy: {html.escape(first["least_squares"]["baseline_mode"])}. Default RAW_SSE uses the exact pointwise equation 9; optional NORMALIZED_MSE gives equal normalized profile weights. UV is predicted as the sum of each species concentration multiplied by its own response factor.</p>
<table><thead><tr><th>Run</th><th>Profile</th><th>Overlap points</th><th>Normalized NRMSE</th><th>Signal RMSE</th><th>Weighted RMSE</th><th>Run weight</th><th>R²</th></tr></thead><tbody>{profile_rows}</tbody></table>
<h2>Measured and predicted chromatograms</h2>{overlays_html}
<h2>Improved shared parameters</h2><p><a href="{FIT_IMPROVED_CSV.name}">Parameter list CSV</a> · <a href="{FIT_IMPROVED_TXT.name}">Parameter list text</a></p><table><thead><tr><th>Parameter</th><th>Initial</th><th>Fitted</th></tr></thead><tbody>{parameter_rows}</tbody></table>
<h2>Profile recipes used</h2>{recipe_details}
<h2>Objective construction</h2><p>Equation 9 is the sum of squared predicted-minus-measured detector samples. RAW_SSE preserves it verbatim. NORMALIZED_MSE divides by each run's signal scale and sample count. WEIGHTED_RMSE is the square root of the run-weighted average of each profile's sample-weighted mean squared error, with independently adjustable positive run weights and optional nonnegative Excel/CSV Weight columns. Minimizing the square root is equivalent to minimizing its squared residual vector with bounded SciPy nonlinear least squares. Baseline estimation is opt-in; the default baseline is fixed. Each LSQ profile has its own independent process recipe (flow, feed, buffers, PLWs, elution and resolution). Only unlocked mechanistic parameters are shared and refined. Column geometry remains fixed.</p>
<h2>Objective improvement over time</h2><p>History file: <a href="{FIT_HISTORY_CSV.name}">{FIT_HISTORY_CSV.name}</a></p>{history_plot}
<p>Files: {FIT_TRACE_CSV.name}, {FIT_RESULTS_CSV.name}, {FIT_COMPOSITION_CSV.name}, {FIT_SETTINGS_JSON.name}.</p>
""", encoding="utf-8")

    return {
        "success": opt_success,
        "accepted": accepted,
        "message": opt_message,
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "improvement_percent": improvement,
        "nfev": opt_nfev,
        "residual_evaluations": eval_count,
        "mode": mode,
        "profile_metrics": metric_rows,
        "initial_weighted_rmse": math.sqrt(max(initial_obj, 0.0)) if weighted_objective else None,
        "final_weighted_rmse": math.sqrt(max(final_obj, 0.0)) if weighted_objective else None,
        "improved_parameters_csv": str(FIT_IMPROVED_CSV),
        "improved_parameters_txt": str(FIT_IMPROVED_TXT),
        "fitted_config": fitted,
        "fit_results_csv": str(FIT_RESULTS_CSV),
        "fit_trace_csv": str(FIT_TRACE_CSV),
        "fit_composition_csv": str(FIT_COMPOSITION_CSV),
        "fit_report_html": str(FIT_REPORT_HTML),
        "fit_config_json": str(FIT_CONFIG_JSON),
        "fit_settings_json": str(FIT_SETTINGS_JSON),
        "fit_history_csv": str(FIT_HISTORY_CSV),
        "latest_fit_history_csv": str(FIT_HISTORY_CSV),
        "apply_jsl": str(FIT_APPLY_JSL),
        "parameters": [spec.path for spec in specs],
    }
