# R22 paper Langmuir equation and buffer salt check

The first attached paper, Mutavdzin and Seidel-Morgenstern (2024), Eq. 4,
defines competitive equilibrium loading as

    q_i = qmax_i * b_i * c_i / (1 + sum_j b_j * c_j), H_i = qmax_i * b_i.

That paper does not put salt in the isotherm. For HIC, this application makes
the local affinity salt dependent while preserving the paper's competitive
denominator and saturation capacity:

    b_eff_i = b_i * exp(k_s_i * local_salt_M)
    q_i = qmax_i * b_eff_i * c_i / (1 + sum_j b_eff_j * c_j).

The salt-response coefficient k_s_i [M^-1] is an empirical parameter for the
particular protein, resin, salt and concentration basis. With k_s_i=0, salt
dependence is intentionally disabled for that component. The load amount can
exceed finite qmax and cause breakthrough even when salt is high. This is not
low-salt elution.

**What a lower %B means.** Buffer A is the 0% B endpoint; Buffer B is the
100% B endpoint. For a volumetric blend, pump salt is

    (1 - %B/100) * resolved Buffer A salt + (%B/100) * resolved Buffer B salt.

For example, with A=0 M and B=2 M, a 10% B STEP sends **0.2 M** to the
mixer, regardless of B remaining at 2 M. At A=B=2 M, the same step sends
**2 M** and cannot cause a salt-gradient elution. Setting only Buffer B to a
higher molarity raises a low-%B blend by that same small fraction. A STEP
uses its **End** set-point; SET_POINT uses **Start**; LINEAR interpolates.

The JMP preview next to each wash/elution stage shows the active pump salt,
the target's b_eff, and whether Buffer A/B or explicit endpoint chemistry is
active. During a transition the binding salt at each column cell follows
mixer and column transport, so it differs briefly from the pump command.
The result CSV/HTML records local outlet binding salt and b_eff; the stage
receipt records programmed endpoints and their selected salt input modes.
SALT_M directly sets analytical salt molarity. In CONDUCTIVITY mode, the
entered conductivity is multiplied by the calibrated factor; if SALT_M is
selected, conductivity is only an optional output overlay.

`python verify_r22_paper_salt_equilibrium.py` checks Eq. 4 for competing
species at zero and nonzero salt, checks the EDM/TDM inversion at extreme
affinity, follows the generated JMP b/k_s sends and saved SALT_M buffer row
through the Python bridge, and simulates both solvers. An undersaturated
load at A=B=2 M gives the same retained chromatogram at 100% and 10% B.
At A=0 M/B=2 M, a 10% B step resolves to 0.2 M and elutes the protein.
These are synthetic checks; native JMP was unavailable for clicking the UI.

## Updating an existing setup

Extract R22 into a new folder. Copy **both** your saved `tdm_inputs.json` and
`tdm_batch_runs.csv` into that folder, then open its
`00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl` and run the selected model. Check
each buffer's **Salt input** mode and resolved salt, the stage pump-salt
preview, and the output's binding salt and affinity. The R21 OneDrive path
startup fix is included.
