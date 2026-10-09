# R17 load-material chemistry and flow overlay

Build `2026-10-07-R17` adds two requested controls to the EDM/TDM Competitive Langmuir and CPA workflow.

## Flow-rate output overlay

The Model Parameters tab now has **Show flow-rate overlay**. The setting is sent through the JMP bridge and saved in `system.show_flow_rate`. Result SVG/HTML output draws the stage flow command on its own right-hand axis; the exported CSV already contains the same value in `flow_mL_min` for every time point. Load, PLW and elution flow changes therefore appear in the trace and the table uses the exact values that drove the transport solver.

## Load material chemistry

Each saved run now has **Load material chemistry**:

- `BUFFER_RECIPE` (default) uses the load Start/End Buffer B % recipe and resolves pH, salt molarity, ionic strength and conductivity from Buffer A/B.
- `EXPLICIT` uses the saved load material source, pH, salt input mode, salt molarity and conductivity fields. It is independent of the load %B display values, so a feed/load material can be entered at its measured pH and conductivity while retaining the %B recipe for reference.

The new selector is saved as `Load_Material_Chemistry_Mode` and is migrated for older CSVs. Older endpoint-chemistry rows remain explicit. In explicit mode, conductivity is converted to analytical salt with the calibrated factor when `CONDUCTIVITY` is selected; `SALT_M` remains authoritative when selected. The resolved load pH, salt, ionic strength and conductivity are recorded in the result receipt. CPA uses pH/ionic strength in the CPA equations; HIC Competitive Langmuir and the HIC CPA extension use analytical salt molarity for the salt-dependent affinity, while pH remains a reported process condition for those HIC isotherms.

Editing a load pH, conductivity, salt input selector or salt value in the JMP panel automatically promotes the saved run to `EXPLICIT`, so the edited value cannot be silently ignored while the load %B recipe remains visible. Manual legacy rows with a populated DIRECT load source and chemistry fields receive the same migration. The regression follows the same JMP field names through the Python bridge, saved run row and effective solver configuration; it changes the conversion factor and `k_s` independently and checks that the resolved salt and UV trace respond.

Focused verification is recorded in `verification/r17_load_material_chemistry.json`: direct load pH 8.25 and conductivity 35 mS/cm reach the load stage as pH 8.25 and salt 0.70 M with a 0.02 M/(mS/cm) factor, even with retained load %B set to 100, and the flow overlay is present in HTML/SVG/CSV.
