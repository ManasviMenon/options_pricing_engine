import numpy as np
import pandas as pd

df = pd.read_csv("data/option_chain.csv")

# Look at the relationship between our_iv and yahoo_iv
valid = df.dropna(subset=["our_iv", "yahoo_iv"])
valid = valid[valid["yahoo_iv"] > 0]

print(f"Contracts: {len(valid)}")
print(f"\nOur IV     — mean: {valid['our_iv'].mean():.4f}, range: [{valid['our_iv'].min():.4f}, {valid['our_iv'].max():.4f}]")
print(f"Yahoo IV   — mean: {valid['yahoo_iv'].mean():.4f}, range: [{valid['yahoo_iv'].min():.4f}, {valid['yahoo_iv'].max():.4f}]")

# Is our IV systematically higher or lower?
diff = valid["our_iv"] - valid["yahoo_iv"]
print(f"\nOur IV minus Yahoo IV:")
print(f"  mean:   {diff.mean():+.4f}  (if consistently one sign, it's systematic)")
print(f"  median: {diff.median():+.4f}")

# Does the difference depend on moneyness or expiry?
print(f"\nDifference by moneyness bucket:")
valid = valid.copy()
valid["m_bucket"] = pd.cut(valid["moneyness"], bins=[0.85, 0.95, 1.0, 1.05, 1.15])
print(valid.groupby("m_bucket", observed=True)["our_iv"].mean().to_string())
print("vs Yahoo:")
print(valid.groupby("m_bucket", observed=True)["yahoo_iv"].mean().to_string())

# Show the 5 worst mismatches
valid["abs_diff"] = np.abs(diff)
print(f"\n5 worst mismatches:")
print(valid.nlargest(5, "abs_diff")[["strike", "T", "moneyness", "market_price", "our_iv", "yahoo_iv"]].to_string())