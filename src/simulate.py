"""
Monte Carlo simulation: draw 10,000 sets of market assumptions, push them all
through the cash-flow engine at once, and summarize the spread of outcomes.

Each random variable, and why it gets this distribution:

  Rent growth     Normal, drawn each year, AR(1) (autocorrelated).
                  Rent growth can be negative, so a normal fits. Markets move in
                  cycles, so a strong year is usually followed by another strong
                  year. Autocorrelation captures that; independent draws would
                  cancel out over 5 years and understate risk.

  Vacancy         Beta, scaled to 3%-12%, drawn once per scenario.
                  Vacancy is bounded: it never drops below frictional turnover
                  (~3%), and in a bad market it spikes well above average but
                  rarely past ~12%. A beta distribution is bounded and right-skewed,
                  so the bad tail is longer than the good one. One draw per
                  scenario = one market environment for the whole hold.

  Expense growth  Normal, drawn each year, independent.
                  Taxes, insurance, and payroll mostly rise with inflation, with
                  year-to-year noise. Narrower than rent growth (1% vs 1.5% std).

  Interest rate   Normal, drawn once per scenario (floor 3%).
  & exit cap      Normal, drawn once per scenario (floor 4%).
                  Drawn together with correlation 0.6: when rates are high, buyers
                  need higher cap rates to make the math work, so exit values fall.
"""

import numpy as np
import pandas as pd

from src.model import project_deal, base_case_inputs


# ---------------------------------------------------------------------------
# Draw the random inputs
# ---------------------------------------------------------------------------

def draw_rent_growth(rng, n, years, mean, std, autocorrelation):
    """AR(1): g_t = mean + phi * (g_{t-1} - mean) + shock.

    The shock is scaled by sqrt(1 - phi^2) so every year still has the target
    std dev. phi = 0 would be independent years; phi = 1 would be one draw
    repeated forever.
    """
    phi = autocorrelation
    growth = np.empty((n, years))
    growth[:, 0] = mean + std * rng.standard_normal(n)
    for t in range(1, years):
        shock = std * np.sqrt(1 - phi ** 2) * rng.standard_normal(n)
        growth[:, t] = mean + phi * (growth[:, t - 1] - mean) + shock
    return growth


def draw_vacancy(rng, n, mean, low, high, concentration):
    """Beta distribution stretched from [0, 1] to [low, high].

    Pick beta shape (a, b) so its mean lands on the target:
      position of mean in the range = (mean - low) / (high - low)
      a = concentration * position, b = concentration - a
    With 6% in a 3-12% range: position = 1/3, so a = 2, b = 4 (right-skewed).
    """
    position = (mean - low) / (high - low)
    a = concentration * position
    b = concentration - a
    return low + (high - low) * rng.beta(a, b, size=n)


def draw_rate_and_cap(rng, n, rate_cfg, cap_cfg, correlation):
    """Two correlated normals.

    z_rate and z_other are independent standard normals. Mixing them as
        z_cap = rho * z_rate + sqrt(1 - rho^2) * z_other
    gives z_cap a correlation of rho with z_rate and still a std dev of 1.
    """
    z_rate = rng.standard_normal(n)
    z_other = rng.standard_normal(n)
    z_cap = correlation * z_rate + np.sqrt(1 - correlation ** 2) * z_other

    rate = rate_cfg["mean"] + rate_cfg["std"] * z_rate
    cap = cap_cfg["mean"] + cap_cfg["std"] * z_cap
    return np.maximum(rate, rate_cfg["floor"]), np.maximum(cap, cap_cfg["floor"])


def draw_inputs(cfg, rng):
    d = cfg["distributions"]
    n = cfg["simulation"]["n_sims"]
    hold = cfg["deal"]["hold_years"]

    rent_growth = draw_rent_growth(rng, n, hold, d["rent_growth"]["mean"],
                                   d["rent_growth"]["std"], d["rent_growth"]["autocorrelation"])
    vacancy = draw_vacancy(rng, n, d["vacancy"]["mean"], d["vacancy"]["min"],
                           d["vacancy"]["max"], d["vacancy"]["concentration"])
    expense_growth = d["expense_growth"]["mean"] + d["expense_growth"]["std"] * rng.standard_normal((n, hold))
    interest_rate, exit_cap = draw_rate_and_cap(rng, n, d["interest_rate"], d["exit_cap"],
                                                d["rate_cap_correlation"])
    return {
        "rent_growth": rent_growth, "vacancy": vacancy, "expense_growth": expense_growth,
        "exit_cap": exit_cap, "interest_rate": interest_rate,
    }


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

def run_simulation(cfg):
    """Returns one row per scenario: the inputs drawn and the resulting returns,
    for both the fixed-rate loan and the floating-rate version."""
    rng = np.random.default_rng(cfg["simulation"]["seed"])
    x = draw_inputs(cfg, rng)
    n = cfg["simulation"]["n_sims"]

    market = dict(rent_growth=x["rent_growth"], vacancy=x["vacancy"],
                  expense_growth=x["expense_growth"], exit_cap=x["exit_cap"])
    fixed = project_deal(cfg, **market, loan_rate=np.full(n, cfg["loan"]["fixed_rate"]))
    floating = project_deal(cfg, **market, loan_rate=x["interest_rate"])

    df = pd.DataFrame({
        "scenario": np.arange(1, n + 1),
        "avg_rent_growth": x["rent_growth"].mean(axis=1),
        "vacancy": x["vacancy"],
        "avg_expense_growth": x["expense_growth"].mean(axis=1),
        "exit_cap": x["exit_cap"],
        "interest_rate": x["interest_rate"],
    })
    for year in range(x["rent_growth"].shape[1]):
        df[f"rent_growth_y{year + 2}"] = x["rent_growth"][:, year]
    df["noi_year1"] = fixed["noi"][:, 0]
    df["noi_exit_year"] = fixed["noi"][:, -1]
    df["sale_price"] = fixed["sale_price"]
    for name, result in [("fixed", fixed), ("floating", floating)]:
        df[f"levered_irr_{name}"] = result["levered_irr"]
        df[f"equity_multiple_{name}"] = result["equity_multiple"]
        df[f"dscr_year1_{name}"] = result["dscr_year1"]
    df["unlevered_irr"] = fixed["unlevered_irr"]   # same for both loans
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

METRICS = {
    "Levered IRR (fixed)": "levered_irr_fixed",
    "Levered IRR (floating)": "levered_irr_floating",
    "Unlevered IRR": "unlevered_irr",
    "Equity multiple (fixed)": "equity_multiple_fixed",
    "Equity multiple (floating)": "equity_multiple_floating",
}


def summary_stats(df):
    rows = []
    for label, col in METRICS.items():
        s = df[col]
        rows.append({
            "Metric": label, "Mean": s.mean(), "Median": s.median(),
            "P5": s.quantile(0.05), "P25": s.quantile(0.25),
            "P75": s.quantile(0.75), "P95": s.quantile(0.95),
        })
    return pd.DataFrame(rows)


def probabilities(df):
    rows = []
    for label, col in [("Fixed-rate loan", "levered_irr_fixed"),
                       ("Floating-rate loan", "levered_irr_floating"),
                       ("Unlevered (no loan)", "unlevered_irr")]:
        irr = df[col]
        rows.append({
            "Case": label,
            "P(IRR < 0%)": (irr < 0).mean(),
            "P(IRR < 10%)": (irr < 0.10).mean(),
            "P(IRR > 15%)": (irr > 0.15).mean(),
        })
    return pd.DataFrame(rows)


DRIVERS = {
    "Exit cap rate": "exit_cap",
    "Rent growth (avg)": "avg_rent_growth",
    "Vacancy": "vacancy",
    "Expense growth (avg)": "avg_expense_growth",
    "Interest rate": "interest_rate",
}


def driver_correlations(df):
    """Rank (Spearman) correlation of each input with levered IRR.
    Unlike the tornado, this picks up indirect effects: the interest rate
    moves the fixed-rate IRR only through its link to the exit cap."""
    ranks = df.rank()
    rows = []
    for label, col in DRIVERS.items():
        rows.append({
            "Driver": label,
            "Corr. w/ IRR (fixed)": ranks[col].corr(ranks["levered_irr_fixed"]),
            "Corr. w/ IRR (floating)": ranks[col].corr(ranks["levered_irr_floating"]),
        })
    return pd.DataFrame(rows)


def tornado(cfg, df, low_pct=0.10, high_pct=0.90):
    """One-at-a-time sensitivity on the fixed-rate base case.

    For each driver, set it to its 10th and 90th percentile from the simulation
    (everything else stays at base case) and record the levered IRR.
    Wider bar = that variable moves IRR more.
    """
    hold = cfg["deal"]["hold_years"]
    base_irr = project_deal(cfg, **base_case_inputs(cfg))["levered_irr"][0]

    rows = []
    for label, col, key, per_year in [
        ("Exit cap rate", "exit_cap", "exit_cap", False),
        ("Rent growth (avg)", "avg_rent_growth", "rent_growth", True),
        ("Vacancy", "vacancy", "vacancy", False),
        ("Expense growth (avg)", "avg_expense_growth", "expense_growth", True),
    ]:
        result = {"Driver": label}
        for side, pct in [("low", low_pct), ("high", high_pct)]:
            value = df[col].quantile(pct)
            inputs = base_case_inputs(cfg)
            inputs[key] = np.full((1, hold), value) if per_year else np.array([value])
            result[f"{side}_input"] = value
            result[f"irr_at_{side}"] = project_deal(cfg, **inputs)["levered_irr"][0]
        result["swing"] = abs(result["irr_at_high"] - result["irr_at_low"])
        rows.append(result)

    out = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    out["base_irr"] = base_irr
    return out
