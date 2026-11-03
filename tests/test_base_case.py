"""
Sanity check: rebuild the base case by hand, using plain arithmetic and
numpy_financial (independent of src/model.py), and confirm the model agrees.

Run:  python tests/test_base_case.py     (or: python -m pytest)
"""

import sys
from pathlib import Path

import numpy_financial as npf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import run_base_case  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "assumptions.yaml"))


def hand_calc():
    price, units = 40_000_000, 200
    loan = 0.65 * price                          # 26,000,000
    equity = price * 1.02 - loan                 # 14,800,000 (closing costs paid with equity)

    # When rent AND expenses both grow 3%, NOI also grows exactly 3%:
    #   Year 1 EGI  = 3.6M x (1 - 6% vacancy + 5% other income) = 3,564,000
    #   Year 1 NOI  = 60% of EGI                                  = 2,138,400
    noi = [2_138_400 * 1.03 ** t for t in range(6)]   # Years 1..6
    capex = units * 300                                # 60,000

    # Monthly mortgage, 6.25%, 30 years
    monthly_pmt = -npf.pmt(0.0625 / 12, 360, loan)
    debt_service = 12 * monthly_pmt
    balance = -npf.fv(0.0625 / 12, 60, -monthly_pmt, loan)   # after 60 payments

    net_sale = noi[5] / 0.0575 * (1 - 0.02)            # Year 6 NOI / exit cap, less 2%

    levered = [-equity] + [noi[t] - capex - debt_service for t in range(5)]
    levered[5] += net_sale - balance
    unlevered = [-price * 1.02] + [noi[t] - capex for t in range(5)]
    unlevered[5] += net_sale

    return {
        "noi_year1": noi[0], "debt_service": debt_service, "balance": balance,
        "levered_irr": npf.irr(levered), "unlevered_irr": npf.irr(unlevered),
        "equity_multiple": sum(levered[1:]) / equity,
    }


def test_base_case_matches_hand_calc():
    hand = hand_calc()
    model = run_base_case(CFG)

    assert abs(model["noi"][0, 0] - hand["noi_year1"]) < 0.01
    assert abs(model["debt_service"][0] - hand["debt_service"]) < 0.01
    assert abs(model["balance_at_exit"][0] - hand["balance"]) < 0.01
    assert abs(model["levered_irr"][0] - hand["levered_irr"]) < 1e-6      # within 0.0001 bp
    assert abs(model["unlevered_irr"][0] - hand["unlevered_irr"]) < 1e-6
    assert abs(model["equity_multiple"][0] - hand["equity_multiple"]) < 1e-9


if __name__ == "__main__":
    test_base_case_matches_hand_calc()
    h = hand_calc()
    print(f"PASS  hand calc levered IRR {h['levered_irr']:.4%}, unlevered IRR "
          f"{h['unlevered_irr']:.4%}, equity multiple {h['equity_multiple']:.4f}x")
