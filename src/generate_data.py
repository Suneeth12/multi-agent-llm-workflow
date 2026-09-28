"""Seeded synthetic sales table the agents analyze."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Widget", "Gadget", "Gizmo", "Doohickey"]


def make(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(REGIONS, n, p=[0.3, 0.2, 0.25, 0.25])
    product = rng.choice(PRODUCTS, n)
    qty = rng.integers(1, 12, n)
    price = rng.choice([9.99, 19.99, 29.99, 49.99], n)
    revenue = (qty * price).round(2)
    margin = rng.uniform(0.15, 0.5, n)
    profit = (revenue * margin).round(2)
    return pd.DataFrame({"region": region, "product": product, "quantity": qty,
                         "price": price, "revenue": revenue, "profit": profit})


def main() -> None:
    DATA.mkdir(exist_ok=True)
    make().to_csv(DATA / "sales.csv", index=False)
    print(f"Wrote sales.csv -> {DATA}")


if __name__ == "__main__":
    main()
