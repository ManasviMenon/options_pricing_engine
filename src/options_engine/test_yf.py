import yfinance as yf

# Grab Apple
ticker = yf.Ticker("AAPL")

# Current spot price
info = ticker.history(period="1d")
spot = float(info["Close"].iloc[-1])
print(f"AAPL spot price: ${spot:.2f}")

# Available expiry dates
expiries = ticker.options
print(f"\nAvailable expiries: {len(expiries)}")
print(f"First few: {expiries[:5]}")

# Grab one option chain (nearest expiry)
if expiries:
    chain = ticker.option_chain(expiries[0])
    print(f"\nCalls for {expiries[0]}: {len(chain.calls)} contracts")
    print(chain.calls[["strike", "lastPrice", "bid", "ask", "impliedVolatility"]].head())