# -*- coding: utf-8 -*-
"""
================================================================================
VERIFY_SECTION5_NUMBERS.PY -- Verify ALL numerical claims in Section 5
(La Paradoja del Directional Accuracy)
================================================================================

Reads pipeline outputs and independently computes every number cited in
Section 5 of the paper, then compares against the stored values.

Checks:
  1. Directional Accuracy (DA) for all 23 models
  2. DA@3x (hit rate on days with position == 3) for selected models
  3. Pearson correlation rho(DA, Return) across all models
  4. Individual return values cited in the prose
  5. % time in 3x position for selected models
  6. % time in cash for LSTM_Attention

Output: results/section5_verification.json

Author: David Gonzalez Canon
================================================================================
"""

import numpy as np
import pickle
import os
import json
from scipy import stats

np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
# paper/scripts/seccion_5_da_paradox/verify_section5_numbers.py
# -> go up 3 levels to repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

BACKTEST_DETAIL_PATH = os.path.join(RESULTS_DIR, "backtest_detail.pkl")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "trained_artifacts.pkl")
BACKTEST_JSON_PATH = os.path.join(RESULTS_DIR, "final_long_only_backtest.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "section5_verification.json")

# Tolerance for floating point comparisons
TOL_DA = 0.002       # 0.2 percentage points for DA
TOL_RETURN = 0.005   # 0.5 percentage points for returns
TOL_CORR = 0.05      # correlation tolerance
TOL_PCT = 0.5        # 0.5 percentage points for % time


def load_data():
    """Load all required pipeline outputs."""
    print("Loading pipeline outputs...")

    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    print("  [OK] trained_artifacts.pkl")

    with open(BACKTEST_DETAIL_PATH, 'rb') as f:
        detail = pickle.load(f)
    print("  [OK] backtest_detail.pkl")

    with open(BACKTEST_JSON_PATH, 'r') as f:
        backtest_json = json.load(f)
    print("  [OK] final_long_only_backtest.json")

    return artifacts, detail, backtest_json


def compute_directional_accuracy(predictions, actual_returns):
    """
    Compute DA: proportion of days where sign(prediction) == sign(actual_return).
    This matches the computation in optimize_and_backtest.py lines 570-572.
    """
    pred_dir = np.sign(predictions)
    actual_dir = np.sign(actual_returns)
    da = float(np.mean(pred_dir == actual_dir))
    return da


def compute_da_at_3x(positions, actual_returns):
    """
    Compute DA@3x: proportion of 3x-position days where market went up.

    DA@3x = sum(pos_t == 3 AND r_mkt > 0) / sum(pos_t == 3)
    """
    mask_3x = (positions == 3)
    n_3x = int(np.sum(mask_3x))
    if n_3x == 0:
        return float('nan'), 0
    hits = np.sum(mask_3x & (actual_returns > 0))
    da_3x = float(hits / n_3x)
    return da_3x, n_3x


def verify_all(artifacts, detail, backtest_json):
    """Compute and verify ALL numerical claims in Section 5."""

    metadata = artifacts['metadata']
    fwd_test = np.array(metadata['forward_returns_test'][:-1])
    n_test = len(fwd_test)

    results = {
        'section': 'Section 5 - La Paradoja del Directional Accuracy',
        'n_test_days': n_test,
        'checks': [],
    }

    all_pass = True

    def add_check(name, computed, paper_value, tolerance, passed, note=""):
        nonlocal all_pass
        if not passed:
            all_pass = False
        entry = {
            'name': name,
            'computed': computed,
            'paper_value': paper_value,
            'tolerance': tolerance,
            'passed': passed,
        }
        if note:
            entry['note'] = note
        results['checks'].append(entry)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: computed={computed}, paper={paper_value}"
              f"{' (' + note + ')' if note else ''}")

    # =========================================================================
    # 1. DIRECTIONAL ACCURACY (DA) FOR ALL MODELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. DIRECTIONAL ACCURACY (DA) FOR ALL MODELS")
    print("=" * 80)

    da_values = {}
    return_values = {}

    for model_name, model_data in artifacts['models'].items():
        test_preds = np.array(model_data['test_predictions']).flatten()
        nt = min(len(test_preds), len(fwd_test))
        test_preds_al = test_preds[:nt]
        fwd_al = fwd_test[:nt]

        # Compute DA independently
        da_computed = compute_directional_accuracy(test_preds_al, fwd_al)
        da_values[model_name] = da_computed

        # Get stored DA from backtest JSON
        if model_name in backtest_json['models']:
            da_stored = backtest_json['models'][model_name]['directional_accuracy']
            diff = abs(da_computed - da_stored)
            passed = diff < 1e-10  # Should be exact match
            add_check(
                f"DA {model_name}",
                round(da_computed, 6),
                round(da_stored, 6),
                1e-10,
                passed,
                f"diff={diff:.2e}"
            )

            # Also store return for correlation
            return_values[model_name] = backtest_json['models'][model_name]['total_return']

    # =========================================================================
    # 2. VERIFY SPECIFIC DA VALUES IN PAPER TABLE (tab:da_paradox)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. VERIFY SPECIFIC DA VALUES IN PAPER TABLE (tab:da_paradox)")
    print("=" * 80)

    # Paper claims (Table tab:da_paradox):
    # GARCH: DA 54.3%
    # Theta: DA 54.6%
    # ExponentialSmoothing: DA 51.3%
    # LSTM_Attention: DA 47.0%
    # Ridge: DA 46.8%
    # CNN_LSTM: DA 45.5%
    paper_da_claims = {
        'GARCH':                 0.543,
        'Theta':                 0.546,
        'ExponentialSmoothing':  0.513,
        'LSTM_Attention':        0.470,
        'Ridge':                 0.468,
        'CNN_LSTM':              0.455,
    }

    for model_name, paper_da in paper_da_claims.items():
        if model_name in da_values:
            computed_da_pct = round(da_values[model_name] * 100, 1)
            paper_da_pct = round(paper_da * 100, 1)
            diff = abs(computed_da_pct - paper_da_pct)
            passed = diff < TOL_DA * 100
            add_check(
                f"Paper DA% {model_name}",
                computed_da_pct,
                paper_da_pct,
                TOL_DA * 100,
                passed,
                f"diff={diff:.1f}pp"
            )

    # =========================================================================
    # 3. DA@3x (HIGH CONVICTION DIRECTIONAL ACCURACY)
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. DA@3x (HIGH CONVICTION DIRECTIONAL ACCURACY)")
    print("=" * 80)

    # Paper claims (Table tab:da_vs_da3x):
    # Model             DA@3x   %Time3x
    # LSTM_Attention     62.3%   12.2%
    # Ridge              58.5%   20.8%
    # RandomForest       51.5%    5.2%
    # CNN_LSTM           58.6%   12.1%
    # GARCH              63.6%    1.7%
    # Theta              53.3%    2.3%
    paper_da3x_claims = {
        'LSTM_Attention': {'da3x': 62.3, 'pct_3x': 12.2},
        'Ridge':          {'da3x': 58.5, 'pct_3x': 20.8},
        'RandomForest':   {'da3x': 51.5, 'pct_3x':  5.2},
        'CNN_LSTM':       {'da3x': 58.6, 'pct_3x': 12.1},
        'GARCH':          {'da3x': 63.6, 'pct_3x':  1.7},
        'Theta':          {'da3x': 53.3, 'pct_3x':  2.3},
    }

    da3x_computed_all = {}

    for model_name, claims in paper_da3x_claims.items():
        if model_name not in detail['models']:
            add_check(
                f"DA@3x {model_name}",
                "N/A",
                claims['da3x'],
                TOL_DA * 100,
                False,
                "Model not found in backtest_detail.pkl"
            )
            continue

        model_detail = detail['models'][model_name]
        positions = np.array(model_detail['positions'])
        # Use actual market returns from artifacts for DA@3x
        nt = len(positions)
        fwd_al = fwd_test[:nt]

        da3x, n_3x = compute_da_at_3x(positions, fwd_al)
        pct_3x = float(np.mean(positions == 3) * 100)

        da3x_computed_all[model_name] = {
            'da3x': round(da3x * 100, 1) if not np.isnan(da3x) else None,
            'n_3x_days': n_3x,
            'pct_3x': round(pct_3x, 1),
        }

        # Check DA@3x
        if not np.isnan(da3x):
            computed_da3x_pct = round(da3x * 100, 1)
            diff_da3x = abs(computed_da3x_pct - claims['da3x'])
            passed_da3x = diff_da3x < TOL_DA * 100
            add_check(
                f"DA@3x {model_name}",
                computed_da3x_pct,
                claims['da3x'],
                TOL_DA * 100,
                passed_da3x,
                f"n_3x_days={n_3x}, diff={diff_da3x:.1f}pp"
            )
        else:
            add_check(
                f"DA@3x {model_name}",
                "NaN (0 days at 3x)",
                claims['da3x'],
                TOL_DA * 100,
                False,
                "No 3x position days"
            )

        # Check %Time 3x
        diff_pct3x = abs(pct_3x - claims['pct_3x'])
        passed_pct3x = diff_pct3x < TOL_PCT
        add_check(
            f"%Time3x {model_name}",
            round(pct_3x, 1),
            claims['pct_3x'],
            TOL_PCT,
            passed_pct3x,
            f"diff={diff_pct3x:.1f}pp"
        )

    # Compute DA@3x for ALL models (for completeness)
    print("\n  --- DA@3x for all models ---")
    for model_name in detail['models']:
        if model_name in da3x_computed_all:
            continue
        model_detail = detail['models'][model_name]
        positions = np.array(model_detail['positions'])
        nt = len(positions)
        fwd_al = fwd_test[:nt]
        da3x, n_3x = compute_da_at_3x(positions, fwd_al)
        pct_3x = float(np.mean(positions == 3) * 100)
        da3x_str = f"{da3x*100:.1f}%" if not np.isnan(da3x) else "N/A"
        print(f"    {model_name:<22} DA@3x={da3x_str:<8} %3x={pct_3x:.1f}% ({n_3x} days)")
        da3x_computed_all[model_name] = {
            'da3x': round(da3x * 100, 1) if not np.isnan(da3x) else None,
            'n_3x_days': n_3x,
            'pct_3x': round(pct_3x, 1),
        }

    # =========================================================================
    # 4. CORRELATION rho(DA, Return) ACROSS ALL 23 MODELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. CORRELATION rho(DA, Return)")
    print("=" * 80)

    model_names_ordered = []
    da_array = []
    ret_array = []

    for model_name in da_values:
        if model_name in return_values:
            model_names_ordered.append(model_name)
            da_array.append(da_values[model_name])
            ret_array.append(return_values[model_name])

    da_array = np.array(da_array)
    ret_array = np.array(ret_array)

    rho, pvalue = stats.pearsonr(da_array, ret_array)
    rho_rounded = round(rho, 2)

    paper_rho = -0.37
    diff_rho = abs(rho_rounded - paper_rho)
    passed_rho = diff_rho < TOL_CORR

    add_check(
        "rho(DA, Return)",
        rho_rounded,
        paper_rho,
        TOL_CORR,
        passed_rho,
        f"exact rho={rho:.4f}, p-value={pvalue:.4f}, diff={diff_rho:.2f}"
    )

    print(f"\n  Model-by-model DA vs Return:")
    # Sort by DA descending for display
    sorted_idx = np.argsort(-da_array)
    for i in sorted_idx:
        print(f"    {model_names_ordered[i]:<22} DA={da_array[i]*100:.1f}%  "
              f"Return={ret_array[i]*100:+.1f}%")

    # =========================================================================
    # 5. INLINE RETURN VALUES CITED IN SECTION 5
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. INLINE RETURN VALUES CITED IN SECTION 5")
    print("=" * 80)

    # Paper claims (from Table tab:da_paradox and prose):
    # GARCH: +27.2%
    # Theta: +44.6%
    # ExponentialSmoothing: -14.5%
    # LSTM_Attention: +400%
    # Ridge: +263%
    # CNN_LSTM: +129%
    # RandomForest: +131%
    paper_return_claims = {
        'GARCH':                 27.2,
        'Theta':                 44.6,
        'ExponentialSmoothing': -14.5,
        'LSTM_Attention':       400.0,
        'Ridge':                263.0,
        'CNN_LSTM':             129.0,
        'RandomForest':         131.0,
    }

    for model_name, paper_ret_pct in paper_return_claims.items():
        if model_name in return_values:
            computed_ret_pct = round(return_values[model_name] * 100, 1)
            # For large returns (>100%), use 1% tolerance
            if abs(paper_ret_pct) > 100:
                tol = 1.0
            else:
                tol = TOL_RETURN * 100
            diff = abs(computed_ret_pct - paper_ret_pct)
            passed = diff < tol
            add_check(
                f"Return {model_name}",
                computed_ret_pct,
                paper_ret_pct,
                tol,
                passed,
                f"diff={diff:.1f}pp"
            )

    # =========================================================================
    # 6. LSTM_ATTENTION SPECIFIC CLAIMS IN PROSE
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. LSTM_ATTENTION SPECIFIC CLAIMS IN PROSE")
    print("=" * 80)

    if 'LSTM_Attention' in detail['models']:
        lstm_detail = detail['models']['LSTM_Attention']
        lstm_positions = np.array(lstm_detail['positions'])
        n_lstm = len(lstm_positions)

        # Paper claims:
        # "permaneciendo en efectivo el 55% del tiempo"
        pct_cash_lstm = float(np.mean(lstm_positions == 0) * 100)
        diff_cash = abs(pct_cash_lstm - 55.0)
        passed_cash = diff_cash < TOL_PCT
        add_check(
            "LSTM_Attention %cash",
            round(pct_cash_lstm, 1),
            55.0,
            TOL_PCT,
            passed_cash,
            f"diff={diff_cash:.1f}pp"
        )

        # "reserva la posicion 3x para el 12.2% del tiempo"
        # Also referenced as "12% de los dias" in later prose
        pct_3x_lstm = float(np.mean(lstm_positions == 3) * 100)
        diff_3x = abs(pct_3x_lstm - 12.2)
        passed_3x = diff_3x < TOL_PCT
        add_check(
            "LSTM_Attention %3x",
            round(pct_3x_lstm, 1),
            12.2,
            TOL_PCT,
            passed_3x,
            f"diff={diff_3x:.1f}pp (prose says '12.2%' and '12%')"
        )

        # "DA general del 47.0%"
        if 'LSTM_Attention' in da_values:
            lstm_da = round(da_values['LSTM_Attention'] * 100, 1)
            diff_da = abs(lstm_da - 47.0)
            passed_da = diff_da < TOL_DA * 100
            add_check(
                "LSTM_Attention DA (prose)",
                lstm_da,
                47.0,
                TOL_DA * 100,
                passed_da,
                f"diff={diff_da:.1f}pp"
            )

        # "DA@3x asciende al 62.3%"
        if 'LSTM_Attention' in da3x_computed_all:
            lstm_da3x = da3x_computed_all['LSTM_Attention']['da3x']
            if lstm_da3x is not None:
                diff_da3x = abs(lstm_da3x - 62.3)
                passed_da3x = diff_da3x < TOL_DA * 100
                add_check(
                    "LSTM_Attention DA@3x (prose)",
                    lstm_da3x,
                    62.3,
                    TOL_DA * 100,
                    passed_da3x,
                    f"diff={diff_da3x:.1f}pp"
                )

    # =========================================================================
    # 7. GARCH SPECIFIC CLAIMS IN PROSE
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. GARCH SPECIFIC CLAIMS IN PROSE")
    print("=" * 80)

    # "GARCH presenta un DA@3x del 63.6% -- el mas alto del estudio"
    # "apenas asigna la posicion 3x el 1.7% del tiempo"
    if 'GARCH' in da3x_computed_all:
        garch_da3x = da3x_computed_all['GARCH']['da3x']
        if garch_da3x is not None:
            diff = abs(garch_da3x - 63.6)
            passed = diff < TOL_DA * 100
            add_check(
                "GARCH DA@3x (prose)",
                garch_da3x,
                63.6,
                TOL_DA * 100,
                passed,
                f"diff={diff:.1f}pp"
            )

    if 'GARCH' in detail['models']:
        garch_positions = np.array(detail['models']['GARCH']['positions'])
        pct_3x_garch = float(np.mean(garch_positions == 3) * 100)
        diff = abs(pct_3x_garch - 1.7)
        passed = diff < TOL_PCT
        add_check(
            "GARCH %3x (prose)",
            round(pct_3x_garch, 1),
            1.7,
            TOL_PCT,
            passed,
            f"diff={diff:.1f}pp"
        )

    # =========================================================================
    # 8. THETA SPECIFIC CLAIMS IN PROSE
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. THETA SPECIFIC CLAIMS IN PROSE")
    print("=" * 80)

    # "Theta (54.6% DA general vs 53.3% DA@3x)"
    if 'Theta' in da_values:
        theta_da = round(da_values['Theta'] * 100, 1)
        diff = abs(theta_da - 54.6)
        passed = diff < TOL_DA * 100
        add_check(
            "Theta DA general (prose)",
            theta_da,
            54.6,
            TOL_DA * 100,
            passed,
            f"diff={diff:.1f}pp"
        )

    if 'Theta' in da3x_computed_all:
        theta_da3x = da3x_computed_all['Theta']['da3x']
        if theta_da3x is not None:
            diff = abs(theta_da3x - 53.3)
            passed = diff < TOL_DA * 100
            add_check(
                "Theta DA@3x (prose)",
                theta_da3x,
                53.3,
                TOL_DA * 100,
                passed,
                f"diff={diff:.1f}pp"
            )

    # =========================================================================
    # 9. VERIFY "DA@3x IS HIGHEST FOR GARCH" CLAIM
    # =========================================================================
    print("\n" + "=" * 80)
    print("9. VERIFY 'GARCH HAS HIGHEST DA@3x' CLAIM")
    print("=" * 80)

    # Filter out models with no 3x days
    models_with_3x = {k: v for k, v in da3x_computed_all.items()
                      if v['da3x'] is not None and v['n_3x_days'] > 0}

    if models_with_3x:
        max_da3x_model = max(models_with_3x.items(), key=lambda x: x[1]['da3x'])
        is_garch_highest = (max_da3x_model[0] == 'GARCH')
        add_check(
            "GARCH has highest DA@3x",
            f"{max_da3x_model[0]} ({max_da3x_model[1]['da3x']}%)",
            "GARCH (63.6%)",
            "N/A",
            is_garch_highest,
            f"All models: " + ", ".join(
                f"{k}={v['da3x']}%" for k, v in
                sorted(models_with_3x.items(), key=lambda x: -x[1]['da3x'])[:5]
            )
        )

    # =========================================================================
    # 10. VERIFY CORRELATION SIGN AND INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("10. VERIFY CORRELATION SIGN AND INTERPRETATION")
    print("=" * 80)

    # "correlacion moderadamente negativa"
    is_negative = rho < 0
    add_check(
        "rho(DA, Return) is negative",
        f"{rho:.4f}",
        "< 0",
        "N/A",
        is_negative,
        "Paper claims 'moderadamente negativa'"
    )

    # Check that high-DA models have lower returns
    # Paper says models with DA > 54% have modest returns
    high_da_models = {n: (da_values[n], return_values[n])
                      for n in da_values
                      if n in return_values and da_values[n] > 0.54}
    if high_da_models:
        avg_return_high_da = np.mean([r for _, r in high_da_models.values()])
        avg_return_all = np.mean(list(return_values.values()))
        high_da_below_avg = avg_return_high_da < avg_return_all
        add_check(
            "High-DA (>54%) models have below-avg return",
            f"avg={avg_return_high_da*100:.1f}%",
            f"< all-model avg ({avg_return_all*100:.1f}%)",
            "N/A",
            high_da_below_avg,
            f"High-DA models: " + ", ".join(
                f"{n}(DA={d*100:.1f}%,Ret={r*100:.1f}%)"
                for n, (d, r) in high_da_models.items()
            )
        )

    # =========================================================================
    # 11. VERIFY RANDOMFOREST DA CLAIM IN TABLE tab:da_vs_da3x
    # =========================================================================
    print("\n" + "=" * 80)
    print("11. VERIFY RANDOMFOREST DA IN TABLE")
    print("=" * 80)

    # Table says RandomForest DA General = 45.9%
    if 'RandomForest' in da_values:
        rf_da = round(da_values['RandomForest'] * 100, 1)
        diff = abs(rf_da - 45.9)
        passed = diff < TOL_DA * 100
        add_check(
            "RandomForest DA General (table)",
            rf_da,
            45.9,
            TOL_DA * 100,
            passed,
            f"diff={diff:.1f}pp"
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_checks = len(results['checks'])
    passed_checks = sum(1 for c in results['checks'] if c['passed'])
    failed_checks = total_checks - passed_checks

    results['summary'] = {
        'total_checks': total_checks,
        'passed': passed_checks,
        'failed': failed_checks,
        'all_pass': all_pass,
    }

    results['computed_values'] = {
        'da_all_models': {k: round(v, 6) for k, v in da_values.items()},
        'da3x_all_models': da3x_computed_all,
        'rho_da_return': {
            'pearson_r': round(float(rho), 4),
            'p_value': round(float(pvalue), 4),
        },
        'return_all_models': {k: round(v, 6) for k, v in return_values.items()},
    }

    print(f"\n  Total checks: {total_checks}")
    print(f"  Passed:       {passed_checks}")
    print(f"  Failed:       {failed_checks}")

    if failed_checks > 0:
        print(f"\n  FAILED CHECKS:")
        for c in results['checks']:
            if not c['passed']:
                print(f"    - {c['name']}: computed={c['computed']}, "
                      f"paper={c['paper_value']}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    return results


def main():
    print("=" * 80)
    print("SECTION 5 VERIFICATION: La Paradoja del Directional Accuracy")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")
    print()

    # Verify paths exist
    for label, path in [("backtest_detail.pkl", BACKTEST_DETAIL_PATH),
                         ("trained_artifacts.pkl", ARTIFACTS_PATH),
                         ("final_long_only_backtest.json", BACKTEST_JSON_PATH)]:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            print(f"\nERROR: Required file not found: {path}")
            print("Run the pipeline first (scripts/optimize_and_backtest.py)")
            return

    artifacts, detail, backtest_json = load_data()
    results = verify_all(artifacts, detail, backtest_json)

    # Save output
    print(f"\nSaving verification results to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [OK] {OUTPUT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
