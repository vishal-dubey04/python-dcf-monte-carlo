# python-dcf-monte-carlo
A production-grade Financial Valuation Engine for AAPL &amp; MSFT using Multi-Stage DCF, Monte Carlo Simulations, and Trading Comps. Built with Python, Pandas, and SciPy.
# Algorithmic Financial Valuation Engine 📈

A comprehensive financial analysis tool built in Python to determine the intrinsic value of publicly traded companies (Apple & Microsoft). This project integrates traditional fundamental analysis with modern data science and statistical risk modeling.

## 🚀 Key Features
- **Automated Data Retrieval:** Real-time extraction of 5-year financial statements via `yfinance`.
- **Multi-Stage DCF Model:** Dynamic Discounted Cash Flow engine with automated assumption tuning based on historical rolling averages.
- **Monte Carlo Simulations:** 10,000-iteration probabilistic modeling of "Fair Value" using SciPy.
- **Scenario Analysis:** Stress-testing across Bull, Base, and Bear cases.
- **Relative Valuation:** Peer-group multiples analysis (P/E, EV/EBITDA, EV/Revenue).
- **Unit Testing:** 37-test suite ensuring 100% calculation accuracy (WACC, CAPM, Gordon Growth).

## 🛠️ Technical Stack
- **Language:** Python 3.x
- **Data:** Pandas, NumPy, YFinance
- **Stats/Math:** SciPy, Statsmodels
- **Visualization:** Matplotlib, Seaborn

## 📊 Sample Results (Microsoft - MSFT)
- **Calculated WACC:** 9.62%
- **DCF Intrinsic Value:** $215.00
- **Monte Carlo Median:** $251.80
- **Probability of Undervaluation:** 3.7%

## 📂 Project Structure
- `main.py`: Core execution script.
- `results/`: Contains generated FCF trajectories, Sensitivity Heatmaps, and Monte Carlo distributions.
- `tests/`: Automated unit testing suite.

## 📝 Disclaimer
This tool is for educational purposes only. Financial investments carry risk; always conduct independent research or consult a certified financial advisor.
