from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def _candidate_variants(raw_value: object) -> list[Path]:
    if raw_value in (None, ""):
        return []
    text = str(raw_value).strip().strip('"').strip("'")
    if not text:
        return []
    strings = [text]
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        uri_path = unquote(parsed.path)
        if parsed.netloc:
            uri_path = f"//{parsed.netloc}{uri_path}"
        strings.append(uri_path)
    # Repair the malformed drive path JMP can pass as C:Users\... .
    if re.match(r"^[A-Za-z]:[^\\/]", text):
        strings.append(text[:2] + os.sep + text[2:])
    if re.match(r"^/[A-Za-z]:[\\/]", text):
        strings.append(text[1:])
    out: list[Path] = []
    seen: set[str] = set()
    for item in strings:
        normalized = item.replace("/", os.sep) if os.name == "nt" else item
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            out.append(Path(normalized).expanduser())
    return out


def resolve_project_dir() -> Path:
    candidates: list[tuple[object, bool]] = [
        (globals().get("tdm_project_dir"), False),
        (globals().get("tdm_project_dir_posix"), False),
        (globals().get("tdm_wrapper_file"), True),
        (os.environ.get("TDM_PROJECT_DIR"), False),
        (Path.cwd(), False),
        (globals().get("__file__"), True),
    ]
    checked: list[str] = []
    for raw, is_file in candidates:
        for candidate in _candidate_variants(raw):
            if is_file:
                candidate = candidate.parent
            try:
                candidate = candidate.resolve()
            except Exception:
                continue
            checked.append(str(candidate))
            if (candidate / "tdm_jmp.py").is_file() and (candidate / "tdm_minimal_model.py").is_file():
                return candidate
    raise RuntimeError("Could not locate the extracted model folder. Checked:\n  " + "\n  ".join(checked))


def _prepare_project_imports(project: Path) -> None:
    """Make this extracted copy authoritative in JMP's persistent Python session."""
    project = project.resolve()
    kept_paths: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or Path.cwd()).expanduser().resolve()
        except Exception:
            kept_paths.append(entry)
            continue
        if resolved != project:
            kept_paths.append(entry)
    sys.path[:] = [str(project), *kept_paths]

    # JMP keeps one Python interpreter alive between Python Submit File calls.
    # Drop modules from a previously opened extraction so their file paths and
    # output constants cannot redirect this UI to another model folder.
    for module_name in (
        "tdm_bridge",
        "tdm_jmp",
        "tdm_least_squares",
        "tdm_minimal_io",
        "tdm_minimal_model",
    ):
        sys.modules.pop(module_name, None)


def main() -> int:
    project = resolve_project_dir()
    os.chdir(project)
    text = str(project)
    _prepare_project_imports(project)
    os.environ["TDM_PROJECT_DIR"] = text
    from tdm_jmp import open_in_jmp
    if not open_in_jmp():
        raise RuntimeError("The JMP interface could not be opened. Check View > Log for the Python/JSL error.")
    return 0


if __name__ in {"__main__", "tdm_setup_launcher"}:
    main()
