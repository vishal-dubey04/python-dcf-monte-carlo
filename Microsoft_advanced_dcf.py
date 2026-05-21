"""
==========================================================================
MICROSOFT CORPORATION (MSFT) — COMPLETE DCF VALUATION MODEL
==========================================================================
Features:
  1. Error handling for missing data (with MSFT fallbacks)
  2. Automated assumption tuning from historical averages
  3. Scenario analysis (Bull / Base / Bear)
  4. Monte Carlo simulation (10,000 runs)
  5. Comparable company analysis (P/E, EV/EBITDA)
  6. Full visualization suite
  7. Unit tests for all calculations (37 tests)

Run in Spyder: Just press F5
==========================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
import time
import os
import traceback

warnings.filterwarnings("ignore")
sns.set_palette("husl")

os.makedirs("results", exist_ok=True)


# ========================================================================
# SECTION 1: CONFIGURATION & MICROSOFT FALLBACK DATA
# ========================================================================

TARGET_TICKER = "MSFT"
COMPANY_NAME = "Microsoft Corporation"
SHORT_NAME = "Microsoft"

# Microsoft relevant peers
PEER_TICKERS = [
    "AAPL",   # Apple
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "NVDA",   # NVIDIA
    "ORCL",   # Oracle
    "CRM",    # Salesforce
    "ADBE"    # Adobe
]

# Microsoft FY2024 Approximate Figures (Fallbacks)
FALLBACK_FINANCIALS = {
    "revenue":            245_122_000_000,
    "cost_of_revenue":     74_114_000_000,
    "gross_profit":       171_008_000_000,
    "operating_income":   109_433_000_000,
    "net_income":          88_136_000_000,
    "ebitda":             135_000_000_000,
    "total_debt":          67_000_000_000,
    "cash":                75_000_000_000,
    "total_assets":       512_000_000_000,
    "total_equity":       268_000_000_000,
    "capex":               44_000_000_000,
    "depreciation":        22_000_000_000,
    "interest_expense":     2_000_000_000,
    "shares_outstanding":   7_430_000_000,
    "current_price":        430.0,
    "market_cap":        3_200_000_000_000,
    "beta":                  1.00,
    "pe_ratio":             36.0,
    "ev_to_ebitda":         25.0,
}


@dataclass
class DCFAssumptions:
    projection_years: int = 5
    revenue_growth_rates: List[float] = field(
        default_factory=lambda: [0.14, 0.13, 0.12, 0.11, 0.10])  # MSFT grows faster than AAPL
    ebitda_margin:  float = 0.45
    tax_rate:       float = 0.18
    capex_pct:      float = 0.18  # MSFT capex intensity is high (data centers)
    depr_pct:       float = 0.09
    nwc_change_pct: float = 0.01
    risk_free_rate:       float = 0.0435
    equity_risk_premium:  float = 0.055
    beta:                 float = 1.00
    cost_of_debt_pretax:  float = 0.035  # MSFT has AAA credit rating
    debt_weight:          float = 0.10   # Very low debt weight
    terminal_growth_rate: float = 0.03
    bull_revenue_bump:  float =  0.02
    bear_revenue_cut:   float = -0.02
    bull_margin_bump:   float =  0.02
    bear_margin_cut:    float = -0.02
    bull_wacc_shift:    float = -0.005
    bear_wacc_shift:    float =  0.01
    mc_revenue_growth_mean:  float = 0.13
    mc_revenue_growth_std:   float = 0.04
    mc_ebitda_margin_mean:   float = 0.45
    mc_ebitda_margin_std:    float = 0.03
    mc_wacc_mean:            float = 0.085
    mc_wacc_std:             float = 0.01
    mc_terminal_growth_mean: float = 0.03
    mc_terminal_growth_std:  float = 0.005


# ========================================================================
# SECTION 2: DATA HANDLER (FIXED NaN HANDLING)
# ========================================================================

class DataHandler:
    def __init__(self, ticker: str = TARGET_TICKER):
        self.ticker_symbol = ticker
        self._ticker = None
        self._info = {}
        self._income = None
        self._balance = None
        self._cashflow = None
        self._errors = []
        self._connect()

    def _connect(self):
        for attempt in range(3):
            try:
                self._ticker = yf.Ticker(self.ticker_symbol)
                self._info = self._ticker.info or {}
                if self._info:
                    print(f"  ✓ Connected to Yahoo Finance ({self.ticker_symbol})")
                    return
            except Exception as exc:
                print(f"  ⚠ Attempt {attempt+1} failed: {exc}")
                time.sleep(2)
        self._errors.append("Could not connect to Yahoo Finance")
        print("  ✗ Using fallback data")

    def _safe_info(self, key: str, fallback_key: str = ""):
        val = self._info.get(key)
        if val is not None:
            return val
        fb_key = fallback_key or key
        fb = FALLBACK_FINANCIALS.get(fb_key)
        if fb is not None:
            self._errors.append(f"info['{key}'] missing → fallback")
        return fb

    def _safe_statement(self, df, row_labels: list, col_idx: int = 0,
                        fallback_key: str = ""):
        if df is not None:
            for label in row_labels:
                try:
                    val = df.loc[label].iloc[col_idx]
                    if pd.notna(val) and val != 0:
                        return float(val)
                except (KeyError, IndexError):
                    continue
        fb = FALLBACK_FINANCIALS.get(fallback_key, 0)
        self._errors.append(f"Row {row_labels[0]} missing → fallback")
        return fb

    def fetch_statements(self):
        if self._ticker is None:
            return
        for name, attr in [
            ("Income statement", "financials"),
            ("Balance sheet", "balance_sheet"),
            ("Cash-flow statement", "cashflow"),
        ]:
            try:
                data = getattr(self._ticker, attr)
                if data is not None and not data.empty:
                    if attr == "financials":
                        self._income = data
                    elif attr == "balance_sheet":
                        self._balance = data
                    else:
                        self._cashflow = data
                    print(f"  ✓ {name}")
                else:
                    print(f"  ⚠ {name} returned empty")
                    self._errors.append(f"{name} empty")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
                self._errors.append(f"{name}: {e}")

    def get_financials(self) -> Dict:
        self.fetch_statements()
        data = {
            "revenue": self._safe_statement(
                self._income,
                ["Total Revenue", "Revenue", "Operating Revenue"],
                fallback_key="revenue"),
            "cost_of_revenue": self._safe_statement(
                self._income,
                ["Cost Of Revenue", "CostOfRevenue"],
                fallback_key="cost_of_revenue"),
            "gross_profit": self._safe_statement(
                self._income,
                ["Gross Profit", "GrossProfit"],
                fallback_key="gross_profit"),
            "operating_income": self._safe_statement(
                self._income,
                ["Operating Income", "OperatingIncome", "EBIT"],
                fallback_key="operating_income"),
            "net_income": self._safe_statement(
                self._income,
                ["Net Income", "NetIncome", "Net Income Common Stockholders"],
                fallback_key="net_income"),
            "ebitda": self._safe_statement(
                self._income,
                ["EBITDA", "Ebitda", "Normalized EBITDA"],
                fallback_key="ebitda"),
            "interest_expense": self._safe_statement(
                self._income,
                ["Interest Expense", "InterestExpense"],
                fallback_key="interest_expense"),
            "total_debt": self._safe_statement(
                self._balance,
                ["Total Debt", "TotalDebt", "Long Term Debt",
                 "Long Term Debt And Capital Lease Obligation"],
                fallback_key="total_debt"),
            "cash": self._safe_statement(
                self._balance,
                ["Cash And Cash Equivalents", "Cash",
                 "Cash Cash Equivalents And Short Term Investments",
                 "CashAndCashEquivalents"],
                fallback_key="cash"),
            "total_assets": self._safe_statement(
                self._balance,
                ["Total Assets", "TotalAssets"],
                fallback_key="total_assets"),
            "total_equity": self._safe_statement(
                self._balance,
                ["Stockholders Equity", "Total Stockholders Equity",
                 "StockholdersEquity",
                 "Total Equity Gross Minority Interest"],
                fallback_key="total_equity"),
            "capex": self._safe_statement(
                self._cashflow,
                ["Capital Expenditure", "CapitalExpenditure",
                 "Capital Expenditures"],
                fallback_key="capex"),
            "depreciation": self._safe_statement(
                self._cashflow,
                ["Depreciation & Amortization",
                 "Depreciation And Amortization",
                 "DepreciationAndAmortization",
                 "Depreciation"],
                fallback_key="depreciation"),
            "shares_outstanding": self._safe_info(
                "sharesOutstanding", "shares_outstanding"),
            "current_price": self._safe_info(
                "currentPrice", "current_price"),
            "market_cap": self._safe_info("marketCap", "market_cap"),
            "beta": self._safe_info("beta", "beta"),
            "pe_ratio": self._safe_info("trailingPE", "pe_ratio"),
            "ev_to_ebitda": self._safe_info(
                "enterpriseToEbitda", "ev_to_ebitda"),
        }
        if data["capex"] is not None and data["capex"] < 0:
            data["capex"] = abs(data["capex"])
        for key, val in data.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                data[key] = FALLBACK_FINANCIALS.get(key, 0)
                self._errors.append(f"'{key}' was None/NaN → fallback")
        return data

    def get_historical_metrics(self, years: int = 4) -> Dict:
        metrics = {
            "revenue_growth_rates": [],
            "ebitda_margins": [],
            "capex_pcts": [],
            "depr_pcts": [],
            "tax_rates": [],
        }
        inc = self._income
        cf = self._cashflow
        if inc is None:
            return metrics
        cols = inc.columns[:years + 1]

        revenues = []
        for c in cols:
            rev = None
            for lbl in ["Total Revenue", "Revenue", "Operating Revenue"]:
                try:
                    v = inc.loc[lbl, c]
                    if pd.notna(v):
                        rev = float(v)
                        break
                except (KeyError, TypeError):
                    continue
            revenues.append(rev)

        for i in range(len(revenues) - 1):
            r1, r2 = revenues[i], revenues[i + 1]
            if r1 and r2 and r2 > 0 and np.isfinite(r1) and np.isfinite(r2):
                g = r1 / r2 - 1
                if np.isfinite(g) and -0.5 < g < 1.0:
                    metrics["revenue_growth_rates"].append(g)

        for c in cols:
            rev = ebitda = None
            for lbl in ["Total Revenue", "Revenue"]:
                try:
                    v = inc.loc[lbl, c]
                    if pd.notna(v):
                        rev = float(v); break
                except (KeyError, TypeError):
                    pass
            for lbl in ["EBITDA", "Ebitda", "Normalized EBITDA"]:
                try:
                    v = inc.loc[lbl, c]
                    if pd.notna(v):
                        ebitda = float(v); break
                except (KeyError, TypeError):
                    pass
            if rev and ebitda and rev > 0 and np.isfinite(rev) and np.isfinite(ebitda):
                m = ebitda / rev
                if 0 < m < 1 and np.isfinite(m):
                    metrics["ebitda_margins"].append(m)

        if cf is not None:
            for c in cf.columns[:years]:
                rev = None
                for lbl in ["Total Revenue", "Revenue"]:
                    try:
                        v = inc.loc[lbl, c]
                        if pd.notna(v):
                            rev = float(v); break
                    except (KeyError, TypeError):
                        pass
                capex = depr = None
                for lbl in ["Capital Expenditure", "CapitalExpenditure"]:
                    try:
                        v = cf.loc[lbl, c]
                        if pd.notna(v):
                            capex = abs(float(v)); break
                    except (KeyError, TypeError):
                        pass
                for lbl in ["Depreciation & Amortization",
                            "Depreciation And Amortization",
                            "Depreciation"]:
                    try:
                        v = cf.loc[lbl, c]
                        if pd.notna(v):
                            depr = float(v); break
                    except (KeyError, TypeError):
                        pass
                if rev and rev > 0 and np.isfinite(rev):
                    if capex and np.isfinite(capex):
                        pct = capex / rev
                        if 0 < pct < 0.5:
                            metrics["capex_pcts"].append(pct)
                    if depr and np.isfinite(depr):
                        pct = depr / rev
                        if 0 < pct < 0.2:
                            metrics["depr_pcts"].append(pct)

        for c in cols:
            pretax = tax = None
            for lbl in ["Pretax Income", "Income Before Tax", "PretaxIncome"]:
                try:
                    v = inc.loc[lbl, c]
                    if pd.notna(v):
                        pretax = float(v); break
                except (KeyError, TypeError):
                    pass
            for lbl in ["Tax Provision", "Income Tax Expense", "TaxProvision"]:
                try:
                    v = inc.loc[lbl, c]
                    if pd.notna(v):
                        tax = float(v); break
                except (KeyError, TypeError):
                    pass
            if pretax and tax and pretax > 0 and np.isfinite(pretax) and np.isfinite(tax):
                rate = tax / pretax
                if 0 <= rate < 0.5 and np.isfinite(rate):
                    metrics["tax_rates"].append(rate)

        return metrics

    def get_peer_data(self, peer_tickers: list) -> pd.DataFrame:
        rows = []
        for t in peer_tickers:
            try:
                info = yf.Ticker(t).info or {}
                rows.append({
                    "Ticker": t,
                    "Name": info.get("shortName", t),
                    "Market_Cap_B": (info.get("marketCap") or 0) / 1e9,
                    "PE_Ratio": info.get("trailingPE"),
                    "Forward_PE": info.get("forwardPE"),
                    "EV_EBITDA": info.get("enterpriseToEbitda"),
                    "EV_Revenue": info.get("enterpriseToRevenue"),
                    "PB_Ratio": info.get("priceToBook"),
                    "Profit_Margin": info.get("profitMargins"),
                    "Revenue_Growth": info.get("revenueGrowth"),
                })
                print(f"  ✓ {t}")
            except Exception as e:
                print(f"  ✗ {t}: {e}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def print_diagnostics(self):
        if self._errors:
            unique = list(set(self._errors))
            print(f"\n  ⚠ {len(unique)} data warnings:")
            for e in unique[:10]:
                print(f"    • {e}")
            if len(unique) > 10:
                print(f"    … and {len(unique)-10} more")
        else:
            print("\n  ✅ All data fetched cleanly")


# ========================================================================
# SECTION 3: DCF ENGINE (FIXED auto_tune NaN handling)
# ========================================================================

class DCFEngine:
    def __init__(self, financials: Dict,
                 assumptions: Optional[DCFAssumptions] = None):
        self.fin = financials
        self.assumptions = assumptions or DCFAssumptions()

    def auto_tune(self, historical: Dict) -> DCFAssumptions:
        a = deepcopy(self.assumptions)

        def _avg(lst, floor=None, cap=None):
            clean = [x for x in lst if x is not None and np.isfinite(x)]
            if not clean:
                return None
            m = float(np.mean(clean))
            if floor is not None:
                m = max(m, floor)
            if cap is not None:
                m = min(m, cap)
            return m

        def _std(lst):
            clean = [x for x in lst if x is not None and np.isfinite(x)]
            if len(clean) < 2:
                return None
            return float(np.std(clean))

        hist_g = _avg(historical.get("revenue_growth_rates", []),
                      floor=-0.05, cap=0.30)
        if hist_g is not None:
            n = a.projection_years
            tg = a.terminal_growth_rate
            a.revenue_growth_rates = [
                round(hist_g - (hist_g - tg) * (i / (n + 2)), 4)
                for i in range(n)
            ]
            a.mc_revenue_growth_mean = hist_g
            std_g = _std(historical.get("revenue_growth_rates", []))
            a.mc_revenue_growth_std = std_g if std_g else 0.04

        hist_m = _avg(historical.get("ebitda_margins", []),
                      floor=0.10, cap=0.60)
        if hist_m is not None:
            a.ebitda_margin = round(hist_m, 4)
            a.mc_ebitda_margin_mean = hist_m
            std_m = _std(historical.get("ebitda_margins", []))
            a.mc_ebitda_margin_std = std_m if std_m else 0.03

        hist_cx = _avg(historical.get("capex_pcts", []),
                       floor=0.05, cap=0.30)
        if hist_cx is not None:
            a.capex_pct = round(hist_cx, 4)

        hist_dp = _avg(historical.get("depr_pcts", []),
                       floor=0.02, cap=0.20)
        if hist_dp is not None:
            a.depr_pct = round(hist_dp, 4)

        hist_tx = _avg(historical.get("tax_rates", []),
                       floor=0.0, cap=0.30)
        if hist_tx is not None:
            a.tax_rate = round(hist_tx, 4)

        if self.fin.get("beta") and np.isfinite(self.fin["beta"]):
            a.beta = self.fin["beta"]

        self.assumptions = a
        return a

    def calculate_wacc(self, override: Optional[float] = None
                       ) -> Tuple[float, float, float]:
        if override is not None:
            return override, override, override
        a = self.assumptions
        ke = a.risk_free_rate + a.beta * a.equity_risk_premium
        kd = a.cost_of_debt_pretax * (1 - a.tax_rate)
        wacc = ke * (1 - a.debt_weight) + kd * a.debt_weight
        return wacc, ke, kd

    def project_fcf(self, growth_rates=None, ebitda_margin=None
                    ) -> pd.DataFrame:
        a = self.assumptions
        rates = growth_rates or a.revenue_growth_rates
        margin = ebitda_margin if ebitda_margin is not None else a.ebitda_margin
        base_rev = self.fin["revenue"]
        rows = []
        for i in range(a.projection_years):
            yr = i + 1
            valid_rates = [r if np.isfinite(r) else 0.10 for r in rates[:yr]]
            rev = base_rev * np.prod([1 + r for r in valid_rates])
            ebitda = rev * margin
            depr = rev * a.depr_pct
            ebit = ebitda - depr
            nopat = ebit * (1 - a.tax_rate)
            capex = rev * a.capex_pct
            nwc = rev * a.nwc_change_pct
            fcf = nopat + depr - capex - nwc
            rows.append({
                "Year": yr,
                "Label": f"FY{2024 + yr}",
                "Revenue": rev,
                "Revenue_Growth": rates[i] if np.isfinite(rates[i]) else 0.10,
                "EBITDA": ebitda,
                "EBITDA_Margin": margin,
                "Depreciation": depr,
                "EBIT": ebit,
                "NOPAT": nopat,
                "CapEx": capex,
                "NWC_Change": nwc,
                "FCF": fcf,
            })
        return pd.DataFrame(rows)

    def terminal_value(self, final_fcf: float, wacc: float,
                       g: Optional[float] = None) -> float:
        g = g or self.assumptions.terminal_growth_rate
        if wacc <= g:
            raise ValueError(
                f"WACC ({wacc:.2%}) must be greater than terminal growth ({g:.2%})"
            )
        return final_fcf * (1 + g) / (wacc - g)

    def run_valuation(self, growth_rates=None, ebitda_margin=None,
                      wacc_override=None, terminal_g=None) -> Dict:
        proj = self.project_fcf(growth_rates, ebitda_margin)
        fcf_values = proj["FCF"].values
        wacc, ke, kd = self.calculate_wacc(wacc_override)
        tv = self.terminal_value(fcf_values[-1], wacc, terminal_g)
        n = self.assumptions.projection_years
        pv_fcfs = [fcf_values[i] / (1 + wacc) ** (i + 1) for i in range(n)]
        pv_tv = tv / (1 + wacc) ** n
        sum_pv = sum(pv_fcfs)
        ev = sum_pv + pv_tv
        net_debt = self.fin["total_debt"] - self.fin["cash"]
        eq_val = ev - net_debt
        per_share = eq_val / self.fin["shares_outstanding"]
        current = self.fin["current_price"]
        upside = (per_share / current - 1) * 100 if current else 0
        return {
            "projections": proj,
            "fcf_values": fcf_values,
            "wacc": wacc,
            "cost_of_equity": ke,
            "cost_of_debt": kd,
            "pv_fcfs": pv_fcfs,
            "sum_pv_fcfs": sum_pv,
            "terminal_value": tv,
            "pv_terminal": pv_tv,
            "enterprise_value": ev,
            "net_debt": net_debt,
            "equity_value": eq_val,
            "value_per_share": per_share,
            "current_price": current,
            "upside_pct": upside,
        }


# ========================================================================
# SECTION 4: SCENARIO ANALYSIS
# ========================================================================

class ScenarioAnalyzer:
    def __init__(self, engine: DCFEngine):
        self.engine = engine
        self.base_a = deepcopy(engine.assumptions)

    def _bump_growth(self, base_rates, bump):
        return [max((r if np.isfinite(r) else 0.10) + bump, -0.10)
                for r in base_rates]

    def run_scenarios(self) -> Dict:
        a = self.base_a
        base_wacc = self.engine.calculate_wacc()[0]
        bull = self.engine.run_valuation(
            growth_rates=self._bump_growth(a.revenue_growth_rates, a.bull_revenue_bump),
            ebitda_margin=min(a.ebitda_margin + a.bull_margin_bump, 0.55),
            wacc_override=max(base_wacc + a.bull_wacc_shift, 0.05),
        )
        bull["label"] = "🐂 Bull"
        base = self.engine.run_valuation()
        base["label"] = "📊 Base"
        bear = self.engine.run_valuation(
            growth_rates=self._bump_growth(a.revenue_growth_rates, a.bear_revenue_cut),
            ebitda_margin=max(a.ebitda_margin + a.bear_margin_cut, 0.25),
            wacc_override=base_wacc + a.bear_wacc_shift,
        )
        bear["label"] = "🐻 Bear"
        return {"bull": bull, "base": base, "bear": bear}

    def summary_table(self, scenarios: Dict) -> pd.DataFrame:
        rows = []
        for key in ["bull", "base", "bear"]:
            s = scenarios[key]
            rows.append({
                "Scenario": s["label"],
                "WACC": f'{s["wacc"]:.2%}',
                "EV ($B)": f'${s["enterprise_value"]/1e9:,.0f}',
                "Equity ($B)": f'${s["equity_value"]/1e9:,.0f}',
                "Value/Share": f'${s["value_per_share"]:,.2f}',
                "Upside": f'{s["upside_pct"]:+.1f}%',
            })
        return pd.DataFrame(rows)


# ========================================================================
# SECTION 5: MONTE CARLO (FIXED NaN handling)
# ========================================================================

class MonteCarloSimulator:
    def __init__(self, engine: DCFEngine,
                 n_simulations: int = 10_000, seed: int = 42):
        self.engine = engine
        self.n = n_simulations
        self.rng = np.random.default_rng(seed)
        self.a = engine.assumptions
        self.results = {}

    def _sample(self, mean, std, low, high, size):
        if not np.isfinite(mean):
            mean = 0.13
        if not np.isfinite(std) or std <= 0:
            std = 0.04
        a_clip = (low - mean) / std
        b_clip = (high - mean) / std
        return stats.truncnorm.rvs(
            a_clip, b_clip, loc=mean, scale=std,
            size=size, random_state=self.rng)

    def run(self) -> Dict:
        a = self.a
        n = self.n
        proj_years = a.projection_years

        rev_growth = self._sample(
            a.mc_revenue_growth_mean, a.mc_revenue_growth_std,
            -0.05, 0.35, n)
        ebitda_margins = self._sample(
            a.mc_ebitda_margin_mean, a.mc_ebitda_margin_std,
            0.30, 0.55, n)
        waccs = self._sample(
            a.mc_wacc_mean, a.mc_wacc_std,
            0.05, 0.15, n)
        terminal_gs = self._sample(
            a.mc_terminal_growth_mean, a.mc_terminal_growth_std,
            0.01, 0.05, n)

        values = np.zeros(n)
        evs = np.zeros(n)
        base_rev = self.engine.fin["revenue"]
        net_debt = self.engine.fin["total_debt"] - self.engine.fin["cash"]
        shares = self.engine.fin["shares_outstanding"]

        for i in range(n):
            g = rev_growth[i]
            margin = ebitda_margins[i]
            wacc = waccs[i]
            tg = terminal_gs[i]
            if wacc <= tg:
                tg = wacc - 0.01
            schedule = [g - (g - tg) * (yr / (proj_years + 2))
                        for yr in range(proj_years)]
            fcfs = []
            rev = base_rev
            for yr in range(proj_years):
                rev *= (1 + schedule[yr])
                ebitda = rev * margin
                depr = rev * a.depr_pct
                ebit = ebitda - depr
                nopat = ebit * (1 - a.tax_rate)
                capex = rev * a.capex_pct
                nwc = rev * a.nwc_change_pct
                fcfs.append(nopat + depr - capex - nwc)
            tv = fcfs[-1] * (1 + tg) / (wacc - tg)
            pv_fcfs = sum(fcfs[j] / (1+wacc)**(j+1) for j in range(proj_years))
            pv_tv = tv / (1+wacc)**proj_years
            ev = pv_fcfs + pv_tv
            evs[i] = ev
            values[i] = (ev - net_debt) / shares

        self.results = {
            "values_per_share": values,
            "enterprise_values": evs,
            "rev_growth": rev_growth,
            "ebitda_margins": ebitda_margins,
            "waccs": waccs,
            "terminal_gs": terminal_gs,
            "n_simulations": n,
        }
        return self.results

    def summary_stats(self) -> pd.DataFrame:
        vps = self.results["values_per_share"]
        current = self.engine.fin["current_price"]
        return pd.DataFrame([{
            "Simulations": f"{self.n:,}",
            "Mean": f"${np.mean(vps):,.2f}",
            "Median": f"${np.median(vps):,.2f}",
            "Std Dev": f"${np.std(vps):,.2f}",
            "5th %ile": f"${np.percentile(vps, 5):,.2f}",
            "25th %ile": f"${np.percentile(vps, 25):,.2f}",
            "75th %ile": f"${np.percentile(vps, 75):,.2f}",
            "95th %ile": f"${np.percentile(vps, 95):,.2f}",
            "P(Undervalued)": f"{np.mean(vps > current):.1%}",
        }])


# ========================================================================
# SECTION 6: COMPARABLE COMPANY ANALYSIS
# ========================================================================

class CompsAnalyzer:
    def __init__(self, company_fin: Dict, peer_data: pd.DataFrame):
        self.fin = company_fin
        self.peers = peer_data.copy() if not peer_data.empty else pd.DataFrame()

    def peer_table(self) -> pd.DataFrame:
        df = self.peers.dropna(subset=["PE_Ratio", "EV_EBITDA"], how="all")
        if df.empty:
            return pd.DataFrame({"Note": ["No peer data available"]})
        target_company = pd.DataFrame([{
            "Ticker": f"{TARGET_TICKER} ★",
            "Name": COMPANY_NAME,
            "Market_Cap_B": self.fin["market_cap"] / 1e9,
            "PE_Ratio": self.fin["pe_ratio"],
            "EV_EBITDA": self.fin["ev_to_ebitda"],
            "Profit_Margin": self.fin["net_income"] / self.fin["revenue"]
                if self.fin["revenue"] else None,
        }])
        combined = pd.concat([target_company, df], ignore_index=True)
        numeric = ["PE_Ratio", "EV_EBITDA"]
        medians = {c: df[c].median() for c in numeric}
        medians["Ticker"] = ""
        medians["Name"] = "── Peer Median ──"
        combined = pd.concat([combined, pd.DataFrame([medians])], ignore_index=True)
        return combined

    def implied_valuations(self) -> pd.DataFrame:
        df = self.peers.dropna(subset=["PE_Ratio", "EV_EBITDA"], how="all")
        if df.empty:
            return pd.DataFrame({"Note": ["No peer data"]})
        shares = self.fin["shares_outstanding"]
        net_debt = self.fin["total_debt"] - self.fin["cash"]
        eps = self.fin["net_income"] / shares if shares else 0
        ebitda = self.fin["ebitda"]
        revenue = self.fin["revenue"]
        current = self.fin["current_price"]
        results = []
        med_pe = df["PE_Ratio"].median()
        if pd.notna(med_pe) and eps > 0:
            price = eps * med_pe
            results.append({
                "Method": "P/E (Trailing)",
                "Peer Median": f"{med_pe:.1f}x",
                "Implied Price": f"${price:,.2f}",
                "vs Current": f"{(price/current - 1)*100:+.1f}%",
            })
        med_ev = df["EV_EBITDA"].median()
        if pd.notna(med_ev) and ebitda > 0:
            implied_ev = ebitda * med_ev
            price = (implied_ev - net_debt) / shares
            results.append({
                "Method": "EV/EBITDA",
                "Peer Median": f"{med_ev:.1f}x",
                "Implied Price": f"${price:,.2f}",
                "vs Current": f"{(price/current - 1)*100:+.1f}%",
            })
        med_rev = df["EV_Revenue"].median()
        if pd.notna(med_rev) and revenue > 0:
            implied_ev = revenue * med_rev
            price = (implied_ev - net_debt) / shares
            results.append({
                "Method": "EV/Revenue",
                "Peer Median": f"{med_rev:.1f}x",
                "Implied Price": f"${price:,.2f}",
                "vs Current": f"{(price/current - 1)*100:+.1f}%",
            })
        return pd.DataFrame(results)


# ========================================================================
# SECTION 7: VISUALIZATIONS
# ========================================================================

def plot_fcf_projections(results):
    proj = results["projections"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(proj))
    w = 0.35
    ax.bar(x - w/2, proj["FCF"]/1e9, w, label="FCF", color="#3498db")
    ax.bar(x + w/2, [v/1e9 for v in results["pv_fcfs"]], w,
           label="PV of FCF", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(proj["Label"])
    ax.set_ylabel("$ Billions")
    ax.set_title("Projected FCF vs Present Value")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    ax.plot(proj["Label"], proj["Revenue"]/1e9, "o-", lw=2, label="Revenue")
    ax.plot(proj["Label"], proj["EBITDA"]/1e9, "s-", lw=2, label="EBITDA")
    ax.fill_between(range(len(proj)), proj["Revenue"]/1e9,
                    proj["EBITDA"]/1e9, alpha=0.15)
    ax.set_ylabel("$ Billions")
    ax.set_title("Revenue & EBITDA Growth")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{SHORT_NAME} — 5-Year Financial Projections", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_valuation_bridge(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["PV of FCFs", "PV Terminal", "Enterprise\nValue",
              "Less Net\nDebt", "Equity\nValue"]
    values = [results["sum_pv_fcfs"]/1e9, results["pv_terminal"]/1e9,
              results["enterprise_value"]/1e9, -results["net_debt"]/1e9,
              results["equity_value"]/1e9]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", lw=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"${val:,.0f}B", ha="center", va="bottom", fontweight="bold",
                fontsize=9)
    ax.set_ylabel("$ Billions")
    ax.set_title("Valuation Bridge", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ev_composition(results):
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [results["sum_pv_fcfs"], results["pv_terminal"]]
    labels = [f'PV of FCFs\n${sizes[0]/1e9:,.0f}B',
              f'PV Terminal\n${sizes[1]/1e9:,.0f}B']
    ax.pie(sizes, labels=labels, autopct="%1.1f%%",
           colors=["#ff9999", "#66b3ff"], startangle=90,
           explode=(0.04, 0))
    ax.set_title("Enterprise Value Composition", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sensitivity(results, financials):
    base_wacc = results["wacc"]
    proj = results["projections"]
    final_fcf = proj.iloc[-1]["FCF"]
    n = len(proj)
    wacc_r = np.arange(max(base_wacc - 0.02, 0.05),
                       base_wacc + 0.03, 0.005)
    growth_r = np.arange(0.015, 0.05, 0.005)
    grid = np.zeros((len(wacc_r), len(growth_r)))
    for i, w in enumerate(wacc_r):
        for j, g in enumerate(growth_r):
            if w <= g:
                grid[i, j] = np.nan
                continue
            tv = final_fcf * (1+g) / (w-g)
            pv_tv = tv / (1+w)**n
            pv_f = sum(proj.iloc[k]["FCF"] / (1+w)**(k+1) for k in range(n))
            ev = pv_f + pv_tv
            eq = ev - (financials["total_debt"] - financials["cash"])
            grid[i, j] = eq / financials["shares_outstanding"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        grid,
        xticklabels=[f"{g:.1%}" for g in growth_r],
        yticklabels=[f"{w:.1%}" for w in wacc_r],
        annot=True, fmt=".0f", cmap="RdYlGn",
        center=financials["current_price"],
        linewidths=0.5, ax=ax)
    ax.set_xlabel("Terminal Growth Rate")
    ax.set_ylabel("WACC")
    ax.set_title(
        f"Sensitivity — Value/Share (current ${financials['current_price']:.0f})",
        fontweight="bold")
    fig.tight_layout()
    return fig


def plot_scenarios(scenarios):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [scenarios[k]["label"] for k in ("bear", "base", "bull")]
    vps = [scenarios[k]["value_per_share"] for k in ("bear", "base", "bull")]
    evs = [scenarios[k]["enterprise_value"]/1e9
           for k in ("bear", "base", "bull")]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    ax = axes[0]
    bars = ax.bar(labels, vps, color=colors, edgecolor="white")
    for b, v in zip(bars, vps):
        ax.text(b.get_x()+b.get_width()/2, v+5, f"${v:,.0f}",
                ha="center", fontweight="bold")
    ax.axhline(scenarios["base"]["current_price"], ls="--", color="black",
               lw=1.2, label=f'Current ${scenarios["base"]["current_price"]:.0f}')
    ax.set_ylabel("Value per Share ($)")
    ax.set_title("Scenario — Value per Share")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    bars = ax.bar(labels, evs, color=colors, edgecolor="white")
    for b, v in zip(bars, evs):
        ax.text(b.get_x()+b.get_width()/2, v+50, f"${v:,.0f}B",
                ha="center", fontweight="bold")
    ax.set_ylabel("Enterprise Value ($B)")
    ax.set_title("Scenario — Enterprise Value")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Bull / Base / Bear Scenario Analysis", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_monte_carlo(mc_results, current_price):
    vps = mc_results["values_per_share"]
    lo, hi = np.percentile(vps, [1, 99])
    vps_plot = vps[(vps >= lo) & (vps <= hi)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(vps_plot, bins=80, color="#3498db", edgecolor="white",
            alpha=0.75, density=True)
    ax.axvline(current_price, color="red", ls="--", lw=2,
               label=f"Current ${current_price:.0f}")
    ax.axvline(np.median(vps), color="green", ls="-", lw=2,
               label=f"Median ${np.median(vps):,.0f}")
    ax.axvline(np.percentile(vps, 5), color="orange", ls=":",
               label=f"5th %ile ${np.percentile(vps, 5):,.0f}")
    ax.axvline(np.percentile(vps, 95), color="orange", ls=":",
               label=f"95th %ile ${np.percentile(vps, 95):,.0f}")
    ax.set_xlabel("Implied Value per Share ($)")
    ax.set_ylabel("Density")
    ax.set_title(f"Monte Carlo — {mc_results['n_simulations']:,} Simulations")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    sample = np.random.choice(len(vps), min(2000, len(vps)), replace=False)
    sc = ax.scatter(mc_results["waccs"][sample]*100,
                    mc_results["rev_growth"][sample]*100,
                    c=vps[sample], cmap="RdYlGn", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label="Value/Share ($)")
    ax.set_xlabel("WACC (%)")
    ax.set_ylabel("Revenue Growth (%)")
    ax.set_title("Input Drivers vs Output Value")
    ax.grid(alpha=0.3)
    fig.suptitle("Monte Carlo Uncertainty Analysis", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_comps(comps_table, peer_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    numeric = comps_table[
        comps_table["Ticker"].apply(
            lambda x: isinstance(x, str) and len(x) > 0)
    ].copy()
    ax = axes[0]
    if "PE_Ratio" in numeric.columns:
        df = numeric.dropna(subset=["PE_Ratio"]).sort_values("PE_Ratio")
        colors = ["#e74c3c" if TARGET_TICKER in str(t) else "#3498db"
                  for t in df["Ticker"]]
        ax.barh(df["Ticker"], df["PE_Ratio"], color=colors, edgecolor="white")
        ax.set_xlabel("P/E Ratio")
        ax.set_title("Trailing P/E — Peer Comparison")
        ax.grid(axis="x", alpha=0.3)
    ax = axes[1]
    if "EV_EBITDA" in numeric.columns:
        df = numeric.dropna(subset=["EV_EBITDA"]).sort_values("EV_EBITDA")
        colors = ["#e74c3c" if TARGET_TICKER in str(t) else "#2ecc71"
                  for t in df["Ticker"]]
        ax.barh(df["Ticker"], df["EV_EBITDA"], color=colors, edgecolor="white")
        ax.set_xlabel("EV / EBITDA")
        ax.set_title("EV/EBITDA — Peer Comparison")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Comparable Company Analysis", fontweight="bold")
    fig.tight_layout()
    return fig


# ========================================================================
# SECTION 8: UNIT TESTS
# ========================================================================

class UnitTests:
    passed = 0
    failed = 0

    @classmethod
    def _assert(cls, condition: bool, test_name: str, detail: str = ""):
        if condition:
            cls.passed += 1
            print(f"   ✅ {test_name}")
        else:
            cls.failed += 1
            print(f"   ❌ {test_name}  →  {detail}")

    @classmethod
    def _make_engine(cls, overrides=None):
        fin = deepcopy(FALLBACK_FINANCIALS)
        if overrides:
            fin.update(overrides)
        return DCFEngine(fin, DCFAssumptions())

    @classmethod
    def test_wacc(cls):
        print("\n  🧪 WACC Calculation Tests:")
        engine = cls._make_engine()
        wacc, ke, kd = engine.calculate_wacc()
        expected_ke = 0.0435 + 1.00 * 0.055
        cls._assert(abs(ke - expected_ke) < 0.001,
                    "Cost of equity matches CAPM formula",
                    f"Expected {expected_ke:.4f}, got {ke:.4f}")
        expected_kd = 0.035 * (1 - 0.18)
        cls._assert(abs(kd - expected_kd) < 0.001,
                    "After-tax cost of debt correct",
                    f"Expected {expected_kd:.4f}, got {kd:.4f}")
        expected_wacc = expected_ke * 0.90 + expected_kd * 0.10
        cls._assert(abs(wacc - expected_wacc) < 0.001,
                    "WACC matches weighted formula",
                    f"Expected {expected_wacc:.4f}, got {wacc:.4f}")
        cls._assert(0.04 < wacc < 0.15, "WACC in reasonable range (4–15%)")
        cls._assert(ke > 0.0435, "Cost of equity exceeds risk-free rate")
        wacc_o, _, _ = engine.calculate_wacc(override=0.10)
        cls._assert(abs(wacc_o - 0.10) < 0.0001, "WACC override works")
        e1 = cls._make_engine(); e1.assumptions.beta = 0.8; w1, _, _ = e1.calculate_wacc()
        e2 = cls._make_engine(); e2.assumptions.beta = 1.2; w2, _, _ = e2.calculate_wacc()
        cls._assert(w2 > w1, "Higher beta → higher WACC")

    @classmethod
    def test_terminal_value(cls):
        print("\n  🧪 Terminal Value Tests:")
        engine = cls._make_engine()
        tv = engine.terminal_value(100, 0.09, 0.03)
        expected = 100 * 1.03 / (0.09 - 0.03)
        cls._assert(abs(tv - expected) < 0.01,
                    f"TV formula correct", f"Expected {expected:.2f}, got {tv:.2f}")
        cls._assert(tv > 0, "Terminal value is positive")
        tv_low = engine.terminal_value(100, 0.09, 0.02)
        tv_high = engine.terminal_value(100, 0.09, 0.04)
        cls._assert(tv_high > tv_low, "Higher growth → higher terminal value")
        error_raised = False
        try:
            engine.terminal_value(100, 0.03, 0.03)
        except ValueError:
            error_raised = True
        cls._assert(error_raised, "ValueError raised when WACC ≤ growth")
        tv_200 = engine.terminal_value(200, 0.09, 0.03)
        cls._assert(abs(tv_200 / tv - 2.0) < 0.001, "TV scales linearly with FCF")

    @classmethod
    def test_projections(cls):
        print("\n  🧪 FCF Projection Tests:")
        engine = cls._make_engine()
        proj = engine.project_fcf()
        cls._assert(len(proj) == engine.assumptions.projection_years,
                    f"Projection has {engine.assumptions.projection_years} years")
        revs = proj["Revenue"].values
        all_growing = all(revs[i] > revs[i-1] for i in range(1, len(revs)))
        cls._assert(all_growing, "Revenue grows each year")
        all_positive = all(proj["FCF"] > 0)
        cls._assert(all_positive, "All projected FCFs are positive")
        margins = (proj["EBITDA"] / proj["Revenue"]).values
        expected_margin = engine.assumptions.ebitda_margin
        margins_correct = all(abs(m - expected_margin) < 0.0001 for m in margins)
        cls._assert(margins_correct, "EBITDA margin applied correctly")
        custom = [0.15] * 5
        proj2 = engine.project_fcf(growth_rates=custom)
        base_rev = engine.fin["revenue"]
        expected_yr1 = base_rev * 1.15
        cls._assert(abs(proj2.iloc[0]["Revenue"] - expected_yr1) < 1000,
                    "Custom growth rates work correctly")

    @classmethod
    def test_full_valuation(cls):
        print("\n  🧪 Full Valuation Tests:")
        engine = cls._make_engine()
        r = engine.run_valuation()
        required = ["projections", "wacc", "pv_fcfs", "terminal_value",
                     "enterprise_value", "equity_value", "value_per_share"]
        all_keys = all(k in r for k in required)
        cls._assert(all_keys, "All required keys present")
        cls._assert(r["value_per_share"] > 0, "Value per share is positive")
        ev_check = abs(r["enterprise_value"] - (r["sum_pv_fcfs"] + r["pv_terminal"]))
        cls._assert(ev_check < 1.0, "EV = sum(PV FCFs) + PV terminal",
                    f"Diff: ${ev_check:,.0f}")
        eq_check = abs(r["equity_value"] - (r["enterprise_value"] - r["net_debt"]))
        cls._assert(eq_check < 1.0, "Equity = EV - net debt")
        v_low = engine.run_valuation(growth_rates=[0.05]*5)["value_per_share"]
        v_high = engine.run_valuation(growth_rates=[0.20]*5)["value_per_share"]
        cls._assert(v_high > v_low, "Higher growth → higher value")
        v_low_wacc = engine.run_valuation(wacc_override=0.07)["value_per_share"]
        v_high_wacc = engine.run_valuation(wacc_override=0.12)["value_per_share"]
        cls._assert(v_low_wacc > v_high_wacc, "Higher WACC → lower value")

    @classmethod
    def test_auto_tune(cls):
        print("\n  🧪 Auto-Tune Tests:")
        engine = cls._make_engine()
        original_margin = engine.assumptions.ebitda_margin
        hist = {
            "revenue_growth_rates": [0.12, 0.15, 0.14],
            "ebitda_margins": [0.44, 0.46, 0.45],
            "capex_pcts": [0.18, 0.19],
            "depr_pcts": [0.09, 0.08],
            "tax_rates": [0.17, 0.18, 0.19],
        }
        tuned = engine.auto_tune(hist)
        cls._assert(abs(tuned.ebitda_margin - np.mean([0.44, 0.46, 0.45])) < 0.01,
                    "EBITDA margin tuned to historical average")
        cls._assert(abs(tuned.tax_rate - np.mean([0.17, 0.18, 0.19])) < 0.01,
                    "Tax rate tuned to historical average")
        engine2 = cls._make_engine()
        orig = engine2.assumptions.ebitda_margin
        tuned2 = engine2.auto_tune({
            "revenue_growth_rates": [],
            "ebitda_margins": [],
            "capex_pcts": [],
            "depr_pcts": [],
            "tax_rates": [],
        })
        cls._assert(tuned2.ebitda_margin == orig, "Empty history keeps defaults")

    @classmethod
    def test_scenarios(cls):
        print("\n  🧪 Scenario Analysis Tests:")
        engine = cls._make_engine()
        sa = ScenarioAnalyzer(engine)
        s = sa.run_scenarios()
        cls._assert("bull" in s and "base" in s and "bear" in s,
                    "All three scenarios present")
        cls._assert(s["bull"]["value_per_share"] > s["base"]["value_per_share"],
                    "Bull > Base value")
        cls._assert(s["base"]["value_per_share"] > s["bear"]["value_per_share"],
                    "Base > Bear value")

    @classmethod
    def test_monte_carlo(cls):
        print("\n  🧪 Monte Carlo Tests:")
        engine = cls._make_engine()
        mc = MonteCarloSimulator(engine, n_simulations=500, seed=42)
        r = mc.run()
        cls._assert(len(r["values_per_share"]) == 500,
                    "Correct number of simulations (500)")
        mean_val = np.mean(r["values_per_share"])
        cls._assert(100 < mean_val < 1000,
                    f"Mean value reasonable (${mean_val:,.0f})")
        cls._assert(np.std(r["values_per_share"]) > 0,
                    "Standard deviation is positive")
        cls._assert(len(r["rev_growth"]) == 500 and len(r["waccs"]) == 500,
                    "Input arrays have correct length")

    @classmethod
    def test_nan_handling(cls):
        print("\n  🧪 NaN Handling Tests:")
        engine = cls._make_engine()
        hist_with_nan = {
            "revenue_growth_rates": [0.12, np.nan, 0.14, None],
            "ebitda_margins": [0.44, np.nan, 0.45],
            "capex_pcts": [np.nan, np.nan],
            "depr_pcts": [],
            "tax_rates": [0.17, np.nan],
        }
        tuned = engine.auto_tune(hist_with_nan)
        cls._assert(np.isfinite(tuned.ebitda_margin),
                    "auto_tune handles NaN in margins")
        cls._assert(np.isfinite(tuned.mc_revenue_growth_mean),
                    "auto_tune handles NaN in growth rates")
        cls._assert(all(np.isfinite(r) for r in tuned.revenue_growth_rates),
                    "Projection growth rates are all finite")

        mc = MonteCarloSimulator(engine, n_simulations=100, seed=42)
        mc.a.mc_revenue_growth_mean = np.nan
        mc.a.mc_revenue_growth_std = np.nan
        r = mc.run()
        cls._assert(np.all(np.isfinite(r["values_per_share"])),
                    "Monte Carlo handles NaN parameters")

    @classmethod
    def run_all(cls):
        cls.passed = 0
        cls.failed = 0
        print("\n" + "="*70)
        print("UNIT TESTS")
        print("="*70)
        cls.test_wacc()
        cls.test_terminal_value()
        cls.test_projections()
        cls.test_full_valuation()
        cls.test_auto_tune()
        cls.test_scenarios()
        cls.test_monte_carlo()
        cls.test_nan_handling()
        print(f"\n  {'='*40}")
        print(f"  Results: {cls.passed} passed, {cls.failed} failed, "
              f"{cls.passed + cls.failed} total")
        print(f"  {'='*40}")
        if cls.failed > 0:
            print("  ⚠ Some tests failed")
        else:
            print("  🎉 ALL TESTS PASSED!")
        return cls.failed == 0


# ========================================================================
# SECTION 9: MAIN
# ========================================================================

def main():
    SEP = "=" * 70
    print(SEP)
    print(f"{COMPANY_NAME.upper()} ({TARGET_TICKER}) — COMPLETE DCF VALUATION")
    print(SEP)

    all_passed = UnitTests.run_all()
    if not all_passed:
        print("\n⚠ Continuing despite test failures...\n")

    print(f"\n{SEP}")
    print(f"📦 STEP 1: Collecting {SHORT_NAME}'s financial data...")
    print(SEP)

    handler = DataHandler(TARGET_TICKER)
    financials = handler.get_financials()
    historical = handler.get_historical_metrics()
    handler.print_diagnostics()

    print(f"\n  Latest revenue  : ${financials['revenue']/1e9:,.1f}B")
    print(f"  Net income      : ${financials['net_income']/1e9:,.1f}B")
    print(f"  EBITDA          : ${financials['ebitda']/1e9:,.1f}B")
    print(f"  Total debt      : ${financials['total_debt']/1e9:,.1f}B")
    print(f"  Cash            : ${financials['cash']/1e9:,.1f}B")
    print(f"  Shares out      : {financials['shares_outstanding']/1e9:.2f}B")
    print(f"  Current price   : ${financials['current_price']:,.2f}")

    print(f"\n{SEP}")
    print("⚙️  STEP 2: Auto-tuning assumptions from history...")
    print(SEP)

    assumptions = DCFAssumptions()
    engine = DCFEngine(financials, assumptions)
    tuned = engine.auto_tune(historical)

    print(f"  Growth schedule : "
          f"{[f'{g:.1%}' if np.isfinite(g) else 'N/A' for g in tuned.revenue_growth_rates]}")
    print(f"  EBITDA margin   : {tuned.ebitda_margin:.1%}")
    print(f"  CapEx %         : {tuned.capex_pct:.2%}")
    print(f"  Depreciation %  : {tuned.depr_pct:.2%}")
    print(f"  Tax rate        : {tuned.tax_rate:.1%}")
    print(f"  Beta            : {tuned.beta:.2f}")

    print(f"\n{SEP}")
    print("💰 STEP 3: Base-case DCF valuation...")
    print(SEP)

    base = engine.run_valuation()

    print(f"\n  WACC              : {base['wacc']:.2%}")
    print(f"  Cost of equity    : {base['cost_of_equity']:.2%}")
    print(f"  Cost of debt (AT) : {base['cost_of_debt']:.2%}")
    print(f"\n  PV of FCFs        : ${base['sum_pv_fcfs']/1e9:,.0f}B")
    print(f"  PV terminal value : ${base['pv_terminal']/1e9:,.0f}B")
    print(f"  Enterprise value  : ${base['enterprise_value']/1e9:,.0f}B")
    print(f"  Net debt          : ${base['net_debt']/1e9:,.0f}B")
    print(f"  Equity value      : ${base['equity_value']/1e9:,.0f}B")
    print(f"\n  ★ Value / share   : ${base['value_per_share']:,.2f}")
    print(f"  ★ Current price   : ${base['current_price']:,.2f}")
    print(f"  ★ Upside          : {base['upside_pct']:+.1f}%")

    proj = base["projections"].copy()
    print(f"\n  5-Year Projections:")
    print(f"  {'Year':<8} {'Revenue':>12} {'EBITDA':>12} {'FCF':>12} {'Growth':>8}")
    print(f"  {'─'*52}")
    for _, row in proj.iterrows():
        g = row['Revenue_Growth']
        g_str = f"{g:>7.1%}" if np.isfinite(g) else "   N/A"
        print(f"  {row['Label']:<8} "
              f"${row['Revenue']/1e9:>10,.1f}B "
              f"${row['EBITDA']/1e9:>10,.1f}B "
              f"${row['FCF']/1e9:>10,.1f}B "
              f"{g_str}")

    print(f"\n{SEP}")
    print("📊 STEP 4: Scenario analysis (Bull / Base / Bear)...")
    print(SEP)

    sa = ScenarioAnalyzer(engine)
    scenarios = sa.run_scenarios()
    scenario_df = sa.summary_table(scenarios)
    print()
    print(scenario_df.to_string(index=False))

    print(f"\n{SEP}")
    print("🎲 STEP 5: Monte Carlo simulation (10,000 runs)...")
    print(SEP)

    mc = MonteCarloSimulator(engine, n_simulations=10_000, seed=42)
    mc_results = mc.run()
    mc_stats = mc.summary_stats()
    print()
    print(mc_stats.T.to_string(header=False))

    print(f"\n{SEP}")
    print("🏢 STEP 6: Comparable company analysis...")
    print(SEP)

    print("\n  Fetching peer data:")
    peer_df = handler.get_peer_data(PEER_TICKERS)
    comps = CompsAnalyzer(financials, peer_df)
    comps_table = comps.peer_table()
    implied = comps.implied_valuations()

    print("\n  Peer Multiples:")
    display_cols = [c for c in ["Ticker", "Name", "PE_Ratio", "EV_EBITDA"]
                    if c in comps_table.columns]
    if display_cols:
        print(comps_table[display_cols].to_string(index=False))

    print("\n  Implied Valuations from Comps:")
    print(implied.to_string(index=False))

    print(f"\n{SEP}")
    print("📈 STEP 7: Generating charts...")
    print(SEP)

    try:
        fig1 = plot_fcf_projections(base)
        fig1.savefig("results/01_fcf_projections.png", dpi=200, bbox_inches="tight")
        print("  ✓ FCF projections")
    except Exception as e:
        print(f"  ✗ FCF projections: {e}")

    try:
        fig2 = plot_valuation_bridge(base)
        fig2.savefig("results/02_valuation_bridge.png", dpi=200, bbox_inches="tight")
        print("  ✓ Valuation bridge")
    except Exception as e:
        print(f"  ✗ Valuation bridge: {e}")

    try:
        fig3 = plot_ev_composition(base)
        fig3.savefig("results/03_ev_composition.png", dpi=200, bbox_inches="tight")
        print("  ✓ EV composition")
    except Exception as e:
        print(f"  ✗ EV composition: {e}")

    try:
        fig4 = plot_sensitivity(base, financials)
        fig4.savefig("results/04_sensitivity.png", dpi=200, bbox_inches="tight")
        print("  ✓ Sensitivity heatmap")
    except Exception as e:
        print(f"  ✗ Sensitivity: {e}")

    try:
        fig5 = plot_scenarios(scenarios)
        fig5.savefig("results/05_scenarios.png", dpi=200, bbox_inches="tight")
        print("  ✓ Scenario comparison")
    except Exception as e:
        print(f"  ✗ Scenarios: {e}")

    try:
        fig6 = plot_monte_carlo(mc_results, financials["current_price"])
        fig6.savefig("results/06_monte_carlo.png", dpi=200, bbox_inches="tight")
        print("  ✓ Monte Carlo distribution")
    except Exception as e:
        print(f"  ✗ Monte Carlo: {e}")

    try:
        if not peer_df.empty:
            fig7 = plot_comps(comps_table, peer_df)
            fig7.savefig("results/07_comps.png", dpi=200, bbox_inches="tight")
            print("  ✓ Comps analysis")
    except Exception as e:
        print(f"  ✗ Comps chart: {e}")

    plt.show()

    print(f"\n{SEP}")
    print("💡 INVESTMENT SUMMARY")
    print(SEP)

    vps = base["value_per_share"]
    cp = base["current_price"]
    mc_vps = mc_results["values_per_share"]

    print(f"\n  Current Price        : ${cp:,.2f}")
    print(f"  DCF Fair Value       : ${vps:,.2f}")
    print(f"  Bull Case            : ${scenarios['bull']['value_per_share']:,.2f}")
    print(f"  Bear Case            : ${scenarios['bear']['value_per_share']:,.2f}")
    print(f"  MC Median            : ${np.median(mc_vps):,.2f}")
    print(f"  MC 5th percentile    : ${np.percentile(mc_vps, 5):,.2f}")
    print(f"  MC 95th percentile   : ${np.percentile(mc_vps, 95):,.2f}")
    print(f"  P(Undervalued)       : {np.mean(mc_vps > cp):.1%}")

    print("\n  Recommendation: ", end="")
    if vps > cp * 1.15:
        print("✅ STRONG BUY — significant margin of safety")
    elif vps > cp * 1.05:
        print("📈 BUY — moderately undervalued")
    elif vps < cp * 0.85:
        print("⛔ SELL — significantly overvalued")
    elif vps < cp * 0.95:
        print("📉 REDUCE — moderately overvalued")
    else:
        print("↔️  HOLD — fairly valued")

    print(f"\n  Charts saved in: {os.path.abspath('results')}/")
    print(SEP)
    print("✅ ANALYSIS COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()