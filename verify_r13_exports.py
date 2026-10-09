"""Confirm exported point residuals and objective contributions match the fit."""
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from verify_r13 import make_hic_config, reference_csv, redirected_fit_outputs
from tdm_minimal_model import simulate
import tdm_least_squares as lsq


def main():
    config = make_hic_config(steps=90, cells=6)
    result = simulate(config)
    receipt = []
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        old = redirected_fit_outputs(folder)
        try:
            path = folder / 'reference.csv'
            reference_csv(path, result, offset=.123)
            for objective in ['RAW_SSE', 'NORMALIZED_MSE']:
                config['least_squares']['objective'] = objective
                single = lsq.run_least_squares_refinement(
                    config, mode='CHROMATOGRAM', chromatogram_csv=path,
                    unlocked_paths=[])
                with Path(single['fit_trace_csv']).open(newline='') as f:
                    rows = list(csv.DictReader(f))
                raw = np.array([float(row['Residual']) for row in rows])
                np.testing.assert_allclose(raw, -.123, atol=1e-7)
                for row in rows:
                    assert abs(float(row['Residual']) -
                               (float(row['Scaled_Simulated_Signal']) -
                                float(row['Measured_Signal']))) < 1e-9

                fit = lsq.run_global_least_squares_refinement(
                    [{'config': config, 'chromatogram_csv': path}],
                    unlocked_paths=[])
                with Path(fit['fit_trace_csv']).open(newline='') as f:
                    rows = list(csv.DictReader(f))
                ref = lsq.load_chromatogram_reference(path, config)
                _, info = lsq._chromatogram_residuals(config, ref, result['trace'])
                scale = info['measured_signal_scale']
                for row in rows:
                    residual = float(row['Residual'])
                    assert abs(residual - (float(row['Predicted_Detector_Signal']) -
                                           float(row['Measured_Signal']))) < 1e-9
                    assert abs(float(row['Normalized_Residual']) - residual / scale) < 1e-12
                contributions = sum(float(row['Profile_Objective_Contribution']) for row in rows)
                assert np.isclose(contributions, fit['final_objective'], rtol=1e-8, atol=1e-12)
                receipt.append({'objective': objective,
                                'exported_contributions': contributions,
                                'reported_objective': fit['final_objective'],
                                'residual_sign': 'predicted - measured',
                                'passed': True})
        finally:
            for name, value in old.items():
                setattr(lsq, name, value)
    Path(__file__).resolve().with_name('verification').joinpath(
        'export_verification.json').write_text(json.dumps(receipt, indent=2))
    print('PASS single/global raw residual signs, normalized exports and objective sums')


if __name__ == '__main__':
    main()
