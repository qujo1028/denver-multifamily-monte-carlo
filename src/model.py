"""
Cash-flow engine for a multifamily acquisition.

Every function here works on numpy arrays, so the SAME code runs:
  - one deterministic base case (arrays with 1 row), and
  - 10,000 Monte Carlo scenarios at once (arrays with 10,000 rows).

Row = one scenario. Column = one year.

Pro forma, top to bottom (per year):
    Gross potential rent (GPR)
  - Vacancy loss                (GPR x vacancy)
  + Other income                (5% of GPR)
  = Effective gross income (EGI)
  - Operating expenses          (40% of EGI in Year 1, then grows at expense growth)
  = Net operating income (NOI)
  - Capex reserve
  = Unlevered cash flow
  - Debt service
  = Levered cash flow (to equity)

Exit (end of Year 5):
    Sale price = Year 6 NOI / exit cap rate
    Net sale   = sale price - selling costs
    Equity gets net sale - remaining loan balance
"""

import numpy as np


# ---------------------------------------------------------------------------
# Loan math
# ---------------------------------------------------------------------------

def annual_debt_service(loan_amount, annual_rate, amort_years, payments_per_year=12):
    """Standard mortgage payment formula, summed over a year.

    payment = L * r / (1 - (1 + r)^-n)
      L = loan amount, r = periodic rate, n = total number of payments
    """
    r = annual_rate / payments_per_year
    n = amort_years * payments_per_year
    payment = loan_amount * r / (1 - (1 + r) ** -n)
    return payment * payments_per_year


def loan_balance(loan_amount, annual_rate, amort_years, years_elapsed, payments_per_year=12):
    """Remaining principal after `years_elapsed` years of payments.

    balance = L(1+r)^k - payment * ((1+r)^k - 1) / r
    i.e. what the loan grew to with interest, minus what the payments paid down.
    """
    r = annual_rate / payments_per_year
    n = amort_years * payments_per_year
    k = years_elapsed * payments_per_year
    payment = loan_amount * r / (1 - (1 + r) ** -n)
    return loan_amount * (1 + r) ** k - payment * ((1 + r) ** k - 1) / r


# ---------------------------------------------------------------------------
# IRR
# ---------------------------------------------------------------------------

def irr(cash_flows, low=-0.99, high=1.0, iterations=100):
    """IRR for many cash-flow rows at once, found by bisection.

    IRR is the discount rate where NPV = 0. For a normal deal (money out
    first, money back later), NPV falls as the rate rises. So:
      - if NPV at the midpoint is still positive, the IRR is higher -> move `low` up
      - otherwise the IRR is lower -> move `high` down
    After 100 halvings the interval is far narrower than a basis point.

    Gives the same answer as numpy_financial.irr (checked in the test),
    but handles all 10,000 scenarios in one vectorized pass.
    """
    cf = np.atleast_2d(cash_flows)
    years = np.arange(cf.shape[1])  # 0, 1, 2, ..., hold
    lo = np.full(cf.shape[0], low)
    hi = np.full(cf.shape[0], high)

    for _ in range(iterations):
        mid = (lo + hi) / 2
        npv = (cf / (1 + mid[:, None]) ** years).sum(axis=1)
        irr_is_higher = npv > 0
        lo = np.where(irr_is_higher, mid, lo)
        hi = np.where(irr_is_higher, hi, mid)

    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# The deal
# ---------------------------------------------------------------------------

def project_deal(cfg, rent_growth, vacancy, expense_growth, exit_cap, loan_rate):
    """Build the full pro forma and return metrics for every scenario.

    Inputs (n = number of scenarios, H = hold years):
      rent_growth     (n, H)  growth INTO years 2..H+1 (column 0 = Year 1 -> Year 2)
      expense_growth  (n, H)  same layout as rent_growth
      vacancy         (n,)    vacancy rate for the whole hold
      exit_cap        (n,)    cap rate applied to Year H+1 NOI at sale
      loan_rate       (n,)    interest rate on the loan
    """
    deal, ops, loan = cfg["deal"], cfg["operations"], cfg["loan"]
    hold = deal["hold_years"]
    n_years = hold + 1                    # need one extra year of NOI for the exit
    n = len(vacancy)

    # --- Revenue -----------------------------------------------------------
    gpr = np.empty((n, n_years))
    gpr[:, 0] = ops["gpr_year1"]
    for t in range(1, n_years):
        gpr[:, t] = gpr[:, t - 1] * (1 + rent_growth[:, t - 1])

    vacancy_loss = gpr * vacancy[:, None]
    other_income = gpr * ops["other_income_pct_of_gpr"]
    egi = gpr - vacancy_loss + other_income

    # --- Expenses ----------------------------------------------------------
    # Year 1 opex is set as a % of EGI. After that it grows at its own rate,
    # so a year of weak rent growth and high expense growth squeezes NOI.
    opex = np.empty((n, n_years))
    opex[:, 0] = egi[:, 0] * ops["opex_pct_of_egi_year1"]
    for t in range(1, n_years):
        opex[:, t] = opex[:, t - 1] * (1 + expense_growth[:, t - 1])

    noi = egi - opex
    capex = deal["units"] * ops["capex_reserve_per_unit"]   # flat $/year

    # --- Purchase & financing ---------------------------------------------
    price = deal["purchase_price"]
    total_cost = price * (1 + deal["closing_cost_pct"])
    loan_amount = price * loan["ltv"]
    equity = total_cost - loan_amount

    debt_service = annual_debt_service(loan_amount, loan_rate, loan["amort_years"],
                                       loan["payments_per_year"])
    balance_at_exit = loan_balance(loan_amount, loan_rate, loan["amort_years"], hold,
                                   loan["payments_per_year"])

    # --- Exit ----------------------------------------------------------------
    sale_price = noi[:, hold] / exit_cap          # Year 6 NOI / exit cap
    net_sale = sale_price * (1 - deal["selling_cost_pct"])

    # --- Cash flows (column 0 = purchase date, columns 1..H = years) -------
    unlevered_cf = np.zeros((n, hold + 1))
    unlevered_cf[:, 0] = -total_cost
    unlevered_cf[:, 1:] = noi[:, :hold] - capex
    unlevered_cf[:, -1] += net_sale

    levered_cf = np.zeros((n, hold + 1))
    levered_cf[:, 0] = -equity
    levered_cf[:, 1:] = noi[:, :hold] - capex - debt_service[:, None]
    levered_cf[:, -1] += net_sale - balance_at_exit

    # --- Return metrics ------------------------------------------------------
    equity_multiple = levered_cf[:, 1:].sum(axis=1) / equity
    dscr_year1 = noi[:, 0] / debt_service

    return {
        "gpr": gpr, "vacancy_loss": vacancy_loss, "other_income": other_income,
        "egi": egi, "opex": opex, "noi": noi, "capex": capex,
        "total_cost": total_cost, "loan_amount": loan_amount, "equity": equity,
        "debt_service": debt_service, "balance_at_exit": balance_at_exit,
        "sale_price": sale_price, "net_sale": net_sale,
        "unlevered_cf": unlevered_cf, "levered_cf": levered_cf,
        "levered_irr": irr(levered_cf),
        "unlevered_irr": irr(unlevered_cf),
        "equity_multiple": equity_multiple,
        "dscr_year1": dscr_year1,
    }


def base_case_inputs(cfg, loan_rate=None):
    """Every random variable set to its mean: the deterministic base case."""
    d = cfg["distributions"]
    hold = cfg["deal"]["hold_years"]
    return {
        "rent_growth": np.full((1, hold), d["rent_growth"]["mean"]),
        "vacancy": np.array([d["vacancy"]["mean"]]),
        "expense_growth": np.full((1, hold), d["expense_growth"]["mean"]),
        "exit_cap": np.array([d["exit_cap"]["mean"]]),
        "loan_rate": np.array([cfg["loan"]["fixed_rate"] if loan_rate is None else loan_rate]),
    }


def run_base_case(cfg):
    """Deterministic run: use this to tie out against an Excel model."""
    return project_deal(cfg, **base_case_inputs(cfg))
