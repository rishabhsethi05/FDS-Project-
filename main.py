import yfinance as yf
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

# User's current portfolio
current_portfolio = {
    "AAPL": 1000,
    "MSFT": 2000,
    "GOOGL": 1500
}

candidate_assets = ["GLD", "VNQ", "BND", "BTC-USD"]  # alternatives

all_assets = list(current_portfolio.keys()) + candidate_assets

# Download price data
data = yf.download(all_assets, start="2018-01-01", end="2024-01-01", auto_adjust=True)["Close"]

# Current portfolio value
current_value = sum(current_portfolio.values())

# Calculate expected returns and covariance
mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Weights for alternative assets only
ef = EfficientFrontier(mu[candidate_assets], S.loc[candidate_assets, candidate_assets])
ef.max_sharpe()
alt_weights = ef.clean_weights()

# Scale weights by desired additional investment amount
additional_investment = 2000  # total amount to add in alternatives
alt_allocation = {asset: weight * additional_investment for asset, weight in alt_weights.items()}

print("Alternative Assets Allocation (Additions):")
for asset, amount in alt_allocation.items():
    print(f"{asset}: ${amount:.2f}")

# Optional: calculate new portfolio performance including added alternatives
new_weights = {**current_portfolio, **alt_allocation}
total_value = current_value + additional_investment
normalized_weights = {k: v/total_value for k,v in new_weights.items()}

ef_full = EfficientFrontier(mu, S)
ef_full.set_weights(list(normalized_weights.values()))
ret, vol, sharpe = ef_full.portfolio_performance()
print(f"\nExpected Return: {ret:.2%}")
print(f"Expected Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
