"""LightGBM quantile regression modellerini egitir (Faz 0 ciktisi).

Calistirma (proje kokunden):
    python -m src.ml.train_delay_model
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src import config

FEATURES = ["early_delay_min", "stations_between", "train_type"]
TARGET = "late_delay_min"


def load_and_clean():
    df = pd.read_parquet(config.TRAINING_PAIRS_PATH)
    print(f"Ham veri: {len(df)} satir")
    before = len(df)
    df = df[
        df["early_delay_min"].between(0, 120)
        & df["late_delay_min"].between(0, 180)
        & (df["delay_change_min"].abs() < 200)
    ]
    print(f"Uc degerler filtrelendi: {before} -> {len(df)} satir")
    df["train_type"] = df["train_type"].astype("category")
    return df


def train_quantile_models(df):
    unique_ids = np.asarray(df["trip_instance_id"].unique(), dtype=object)
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_df = df[df["trip_instance_id"].isin(train_ids)]
    test_df = df[df["trip_instance_id"].isin(test_ids)]
    print(f"Egitim: {len(train_df)} satir | Test: {len(test_df)} satir (yolculuk bazinda ayrildi)")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    categories = df["train_type"].cat.categories

    models = {}
    for q in (0.1, 0.5, 0.9):
        print(f"\nq={q} modeli egitiliyor...")
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=200,
            num_leaves=31, learning_rate=0.05, verbose=-1,
        )
        model.fit(X_train, y_train, categorical_feature=["train_type"])
        models[q] = model
        coverage = (y_test <= model.predict(X_test)).mean()
        print(f"  test kapsam orani: {coverage:.2%} (hedef {q:.0%})")

    return models, categories


def main():
    config.ensure_dirs()
    df = load_and_clean()
    models, categories = train_quantile_models(df)

    print("\n" + "=" * 60)
    print("Ornek: ICE, 1 durak sonrasi, su an 6 dakika gecikme")
    print("=" * 60)
    example = pd.DataFrame({
        "early_delay_min": [6],
        "stations_between": [1],
        "train_type": pd.Categorical(["ICE"], categories=categories),
    })
    for q, model in models.items():
        print(f"  q={q}: tahmini varis gecikmesi = {model.predict(example)[0]:.1f} dk")

    for q, model in models.items():
        path = config.MODEL_PATHS[q]
        model.booster_.save_model(str(path))
        print(f"Kaydedildi -> {path}")


if __name__ == "__main__":
    main()
