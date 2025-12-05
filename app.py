import streamlit as st
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer – Alternative Assets Addition")
st.write("Optimize your portfolio by adding alternative assets without changing your existing holdings.")

st.sidebar.header("Current Portfolio")
tickers_input = st.sidebar.text_area("Enter US stock tickers (comma-separated)", "AAPL,MSFT,GOOGL")
amounts_input = st.sidebar.text_area("Enter invested amounts (comma-separated)", "1000,2000,1500")
additional_investment = st.sidebar.number_input("Amount to invest in alternatives ($)", min_value=0, value=2000)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
amounts = [float(a.strip()) for a in amounts_input.split(",") if a.strip()]
current_portfolio = dict(zip(tickers, amounts))

st.sidebar.header("Alternative Assets")
all_alternatives = [
    "GLD", "SLV", "VNQ", "BND", "AGG", "BTC-USD", "ETH-USD", "LQD", "TLT", "GSG", "DBC"
]

include_all = st.sidebar.checkbox("Include all alternative assets", value=True)
if include_all:
    candidate_assets = all_alternatives
else:
    candidate_assets = st.sidebar.multiselect(
        "Select alternative assets to consider",
        all_alternatives,
        default=["GLD", "VNQ", "BND", "BTC-USD"]
    )

def validate_tickers(ticker_list):
    valid = []
    invalid = []
    for t in ticker_list:
        try:
            data = yf.download(t, period="1mo", progress=False)
            if not data.empty:
                valid.append(t)
            else:
                invalid.append(t)
        except:
            invalid.append(t)
    return valid, invalid

if st.button("Optimize Portfolio"):
    if not tickers or not amounts or not candidate_assets:
        st.error("Please enter your portfolio and select at least one alternative asset.")
    else:
        valid_tickers, invalid_tickers = validate_tickers(tickers)
        if invalid_tickers:
            st.warning(f"The following tickers are invalid or not found: {', '.join(invalid_tickers)}")
        if not valid_tickers:
            st.error("No valid tickers to process.")
        else:
            all_assets = valid_tickers + candidate_assets
            data = yf.download(all_assets, start="2018-01-01", end="2024-01-01", auto_adjust=True)["Close"]

            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)

            ef = EfficientFrontier(mu[candidate_assets], S.loc[candidate_assets, candidate_assets])
            ef.max_sharpe()
            alt_weights = ef.clean_weights()
            alt_allocation = {asset: weight * additional_investment for asset, weight in alt_weights.items()}

            st.subheader("Recommended Alternative Assets Allocation")
            st.table(pd.DataFrame(list(alt_allocation.items()), columns=["Asset", "Amount ($)"]))

            total_portfolio = {**{t: current_portfolio[t] for t in valid_tickers}, **alt_allocation}
            total_value = sum(total_portfolio.values())
            normalized_weights = {k: v/total_value for k,v in total_portfolio.items()}

            ef_full = EfficientFrontier(mu, S)
            ef_full.set_weights(normalized_weights)
            ret, vol, sharpe = ef_full.portfolio_performance()

            st.subheader("Expected Portfolio Performance After Addition")
            st.write(f"Expected Return: {ret:.2%}")
            st.write(f"Expected Volatility: {vol:.2%}")
            st.write(f"Sharpe Ratio: {sharpe:.2f}")

            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(normalized_weights.values()), labels=list(normalized_weights.keys()), autopct='%1.1f%%')
            ax.set_title("Portfolio Allocation After Adding Alternatives")
            st.pyplot(fig)
