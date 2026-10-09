"""Reproduce the OneDrive folder-name false positive in the JMP launcher."""
from __future__ import annotations

import tempfile
from pathlib import Path

import tdm_jmp
from tdm_minimal_io import CONFIG_PATH, RESULT_GUARD_BUILD, load_config


WINDOWS_FOLDER = (
    "C:/Users/etm50491/OneDrive - Groupe IPSEN/Documents/JMP Models/"
    "No Model Template 4.0/TDM22_HIC_LSQ_R21"
)


def main() -> None:
    old_path = tdm_jmp._path
    old_available = tdm_jmp.JMP_AVAILABLE
    old_jmp = tdm_jmp.jmp
    old_generated = tdm_jmp.LAST_JSL
    submitted: list[str] = []

    class DesktopStub:
        @staticmethod
        def run_jsl(script: str) -> None:
            submitted.append(script)

    try:
        # On Windows, all absolute file paths passed through _path() contain
        # the parent's "No Model Template 4.0" text. Stub only that platform
        # conversion and the native submission; exercise open_in_jmp itself.
        tdm_jmp._path = lambda path: tdm_jmp._q(WINDOWS_FOLDER + "/" + Path(path).name)
        tdm_jmp.JMP_AVAILABLE = True
        tdm_jmp.jmp = DesktopStub()
        with tempfile.TemporaryDirectory() as directory:
            tdm_jmp.LAST_JSL = Path(directory) / "last_generated_ui.jsl"
            assert tdm_jmp.open_in_jmp()
            script = submitted[0]
            assert WINDOWS_FOLDER in script and RESULT_GUARD_BUILD in script
            assert tdm_jmp.LAST_JSL.read_text(encoding="utf-8") == script

        # The guard still rejects an actual unwanted model choice.
        bad = script.replace('modelBox = Combo Box({', 'modelBox = Combo Box({"SMA", ', 1)
        try:
            tdm_jmp._validate_jsl(bad)
        except ValueError as exc:
            assert "unsupported choices" in str(exc), str(exc)
        else:
            raise AssertionError("The model selector accepted SMA.")

        # It also rejects a removed UI label, while allowing those words in
        # the directory string used for the bridge paths.
        bad = script.replace('Text Box("Model combination"', 'Text Box("Model template"', 1)
        try:
            tdm_jmp._validate_jsl(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("The UI validator accepted a template label.")
    finally:
        tdm_jmp._path = old_path
        tdm_jmp.JMP_AVAILABLE = old_available
        tdm_jmp.jmp = old_jmp
        tdm_jmp.LAST_JSL = old_generated
    print("PASS R21: OneDrive 'No Model Template 4.0' path opens the generated UI; obsolete model controls remain rejected")


if __name__ == "__main__":
    main()
