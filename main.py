"""
Run everything:
  1. Deterministic base case (tie this out to Excel)
  2. 10,000-scenario Monte Carlo simulation
  3. Summary stats, probabilities, tornado
  4. CSV + Excel exports and PNG charts in outputs/

Usage:  python main.py            (uses assumptions.yaml)
        python main.py my_deal.yaml
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.model import run_base_case
from src.simulate import run_simulation, summary_stats, probabilities, tornado, driver_correlations
from src import charts

OUT = Path("outputs")


def load_config(path="assumptions.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def base_case_table(cfg, r):
    """Annual pro forma for the base case, laid out like an Excel model."""
    hold = cfg["deal"]["hold_years"]
    years = [f"Year {t}" for t in range(hold + 2)]  # Year 0 .. Year 6
    blank = [None]

    def ops_row(values):        # Years 1..6 (Year 6 shown for the exit NOI)
        return blank + list(values)

    def cf_row(values):         # Years 0..5
        return list(values) + blank

    rows = {
        "Gross potential rent": ops_row(r["gpr"][0]),
        "Less: vacancy": ops_row(-r["vacancy_loss"][0]),
        "Plus: other income": ops_row(r["other_income"][0]),
        "Effective gross income": ops_row(r["egi"][0]),
        "Less: operating expenses": ops_row(-r["opex"][0]),
        "Net operating income": ops_row(r["noi"][0]),
        "Less: capex reserve": cf_row([None] + [-r["capex"]] * hold),
        "Less: debt service": cf_row([None] + [-r["debt_service"][0]] * hold),
        "Purchase + closing costs": cf_row([-r["total_cost"]] + [None] * hold),
        "Loan proceeds": cf_row([r["loan_amount"]] + [None] * hold),
        "Net sale proceeds": cf_row([None] * hold + [r["net_sale"][0]]),
        "Loan payoff": cf_row([None] * hold + [-r["balance_at_exit"][0]]),
        "Unlevered cash flow": cf_row(r["unlevered_cf"][0]),
        "Levered cash flow": cf_row(r["levered_cf"][0]),
    }
    table = pd.DataFrame(rows, index=years).T

    metrics = pd.DataFrame({"Value": {
        "Purchase price": cfg["deal"]["purchase_price"],
        "Loan amount": r["loan_amount"],
        "Equity invested": r["equity"],
        "Going-in cap rate (Y1 NOI / price)": r["noi"][0, 0] / cfg["deal"]["purchase_price"],
        "Year 1 DSCR": r["dscr_year1"][0],
        "Gross sale price": r["sale_price"][0],
        "Loan balance at exit": r["balance_at_exit"][0],
        "Levered IRR": r["levered_irr"][0],
        "Unlevered IRR": r["unlevered_irr"][0],
        "Equity multiple": r["equity_multiple"][0],
    }})
    return table, metrics


def flatten(cfg, prefix=""):
    """Nested yaml -> flat rows for the Assumptions sheet."""
    rows = []
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows += flatten(value, name + ".")
        else:
            rows.append({"Assumption": name, "Value": value})
    return rows


def write_excel(path, sheets, pct_sheets):
    """Write each DataFrame to its own sheet with readable number formats."""
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, (df, keep_index) in sheets.items():
            df.to_excel(xl, sheet_name=name, index=keep_index)
            ws = xl.sheets[name]
            ws.column_dimensions["A"].width = 38
            for col in ws.iter_cols(min_col=2):
                ws.column_dimensions[col[0].column_letter].width = 18
                for cell in col[1:]:
                    if not isinstance(cell.value, (int, float)):
                        continue
                    if name in pct_sheets or (abs(cell.value) < 1 and cell.value != 0):
                        cell.number_format = "0.00%"
                    elif abs(cell.value) < 10:
                        cell.number_format = "0.00"
                    else:
                        cell.number_format = "#,##0;(#,##0)"


def main(config_path="assumptions.yaml"):
    cfg = load_config(config_path)
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    # 1. Base case ------------------------------------------------------------
    base = run_base_case(cfg)
    table, base_metrics = base_case_table(cfg, base)
    print("\n=== BASE CASE (every input at its mean) ===")
    print(table.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}").to_string())
    print()
    for k, v in base_metrics["Value"].items():
        print(f"  {k:<38} {v:,.4f}" if abs(v) < 10 else f"  {k:<38} {v:,.0f}")

    # 2. Simulation -----------------------------------------------------------
    df = run_simulation(cfg)
    stats = summary_stats(df)
    probs = probabilities(df)
    torn = tornado(cfg, df)
    corr = driver_correlations(df)

    print(f"\n=== MONTE CARLO ({len(df):,} scenarios, seed {cfg['simulation']['seed']}) ===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(probs.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
    print("\nTornado (levered IRR, fixed loan, inputs at P10/P90):")
    print(torn[["Driver", "low_input", "high_input", "irr_at_low", "irr_at_high", "swing"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2%}"))
    print("\nRank correlation of each input with levered IRR:")
    print(corr.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    # 3. Exports --------------------------------------------------------------
    df.to_csv(OUT / "simulation_results.csv", index=False, float_format="%.6f")
    write_excel(OUT / "summary.xlsx", {
        "Summary Stats": (stats.set_index("Metric"), True),
        "Probabilities": (probs.set_index("Case"), True),
        "Base Case Pro Forma": (table, True),
        "Base Case Metrics": (base_metrics, True),
        "Tornado": (torn.set_index("Driver"), True),
        "Driver Correlations": (corr.set_index("Driver"), True),
        "Assumptions": (pd.DataFrame(flatten(cfg)).set_index("Assumption"), True),
    }, pct_sheets={"Probabilities", "Tornado"})
    paths = charts.make_all(df, torn, OUT / "charts")

    print(f"\nSaved: {OUT / 'simulation_results.csv'}, {OUT / 'summary.xlsx'}")
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main(*sys.argv[1:])
