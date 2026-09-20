# src/generate_synthetic_data.py

from pathlib import Path

import numpy as np
import pandas as pd


def generate_vehicle_data(n=2000, random_state=42):
    """Generate synthetic used-vehicle data."""

    rng = np.random.default_rng(random_state)

    # Vehicle attributes
    makes = [
        "Toyota",
        "Honda",
        "Nissan",
        "Mazda",
        "Ford"
    ]

    models = {
        "Toyota": [
            "Corolla",
            "Camry",
            "RAV4",
            "Highlander"
        ],
        "Honda": [
            "Civic",
            "Accord",
            "CR-V"
        ],
        "Nissan": [
            "Sentra",
            "Altima",
            "Rogue"
        ],
        "Mazda": [
            "Mazda3",
            "CX-5",
            "CX-9"
        ],
        "Ford": [
            "Focus",
            "Escape",
            "Explorer"
        ]
    }

    model_base_price = {
        "Corolla": 18500,
        "Camry": 22000,
        "RAV4": 26000,
        "Highlander": 34000,
        "Civic": 19500,
        "Accord": 22500,
        "CR-V": 27000,
        "Sentra": 17000,
        "Altima": 20000,
        "Rogue": 24500,
        "Mazda3": 20000,
        "CX-5": 25500,
        "CX-9": 32000,
        "Focus": 17500,
        "Escape": 23500,
        "Explorer": 33000
    }

    engine_options = [
        "1.5L",
        "1.8L",
        "2.0L",
        "2.4L",
        "2.5L",
        "3.0L",
        "3.5L"
    ]

    transmission_options = [
        "Automatic",
        "Manual"
    ]

  # Generate basic vehicle attributes
    make = rng.choice(
        makes,
        size=n,
        p=[
            0.40,
            0.20,
            0.15,
            0.10,
            0.15
        ]
    )

    model = [
        rng.choice(models[m])
        for m in make
    ]

    year = rng.integers(
        2016,
        2027,
        size=n
    )

    engine = rng.choice(
        engine_options,
        size=n,
        p=[
            0.10,
            0.20,
            0.25,
            0.15,
            0.15,
            0.10,
            0.05
        ]
    )

    transmission = rng.choice(
        transmission_options,
        size=n,
        p=[
            0.85,
            0.15
        ]
    )

    # --------------------------------------------------
    # Mileage
    # --------------------------------------------------

    vehicle_age = 2026 - year

    # Average mileage increases with vehicle age
    base_mileage = (
        vehicle_age * 10500
    )

    mileage = (
        base_mileage
        + rng.normal(
            0,
            12000,
            size=n
        )
    )

    mileage = np.clip(
        mileage,
        500,
        180000
    ).astype(int)

    # Price calculation

    # Base model value
    base_price = np.array([
        model_base_price[m]
        for m in model
    ])

    # Age depreciation
    age_effect = (
        vehicle_age * 1800
    )

    # Mileage depreciation
    mileage_effect = (
        mileage * 0.075
    )

    # Engine effect
    engine_effect = {
        "1.5L": 0,
        "1.8L": 500,
        "2.0L": 1000,
        "2.4L": 1500,
        "2.5L": 2000,
        "3.0L": 3500,
        "3.5L": 4500
    }

    engine_effect = np.array([
        engine_effect[e]
        for e in engine
    ])

    # Transmission effect
    transmission_effect = np.where(
        transmission == "Automatic",
        1200,
        0
    )

    # Small random market effect
    market_noise = rng.normal(
        0,
        1800,
        size=n
    )

    # Final price
    price = (
        base_price
        - age_effect
        - mileage_effect
        + engine_effect
        + transmission_effect
        + market_noise
    )

    # Prevent unrealistic prices
    price = np.clip(
        price,
        3000,
        45000
    )

    price = np.round(
        price,
        2
    )


    # Create df
    df = pd.DataFrame({
        "vehicle.make": make,
        "vehicle.model": model,
        "vehicle.year": year,
        "vehicle.engine": engine,
        "vehicle.transmission": transmission,
        "retailListing.miles": mileage,
        "retailListing.price": price
    })

    return df


if __name__ == "__main__":

    df = generate_vehicle_data(
        n=2000
    )

    output_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "synthetic_vehicle_data.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Generated {len(df):,} synthetic vehicle records."
    )

    print(
        f"Saved to: {output_path}"
    )

    print("\nSample data:")
    print(
        df.head()
    )

    print("\nPrice statistics:")
    print(
        df["retailListing.price"].describe()
    )