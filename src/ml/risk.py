"""Egitilmis modelleri yukler ve aktarma kacirma olasiligini hesaplar."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from src import config

_models = None
_categories = None


def _load():
    global _models, _categories
    if _models is None:
        missing = [p for p in config.MODEL_PATHS.values() if not p.exists()]
        if missing:
            raise SystemExit(
                "Model dosyalari bulunamadi: " + ", ".join(str(p) for p in missing)
                + "\nOnce 'python -m src.ml.train_delay_model' calistir."
            )
        _models = {q: lgb.Booster(model_file=str(p)) for q, p in config.MODEL_PATHS.items()}
        _categories = (
            pd.read_parquet(config.TRAINING_PAIRS_PATH)["train_type"]
            .astype("category").cat.categories
        )
    return _models, _categories


def assess(current_delay_min: float, train_type: str,
           stations_between: int = None, buffer_time_min: float = None) -> dict:
    models, categories = _load()
    stations_between = stations_between or config.STATIONS_BETWEEN
    buffer_time_min = buffer_time_min or config.BUFFER_TIME_MIN

    # Egitimde gorulmemis bir tren tipi gelirse ilk kategoriye dus.
    safe_type = train_type if train_type in categories else categories[0]
    example = pd.DataFrame({
        "early_delay_min": [current_delay_min],
        "stations_between": [stations_between],
        "train_type": pd.Categorical([safe_type], categories=categories),
    })

    preds = {q: float(m.predict(example)[0]) for q, m in models.items()}
    p_miss = float(np.clip(
        1 - np.interp(buffer_time_min, [preds[0.1], preds[0.5], preds[0.9]], [0.1, 0.5, 0.9]),
        0, 1,
    ))
    return {
        "q10": round(preds[0.1], 1),
        "q50": round(preds[0.5], 1),
        "q90": round(preds[0.9], 1),
        "p_miss": round(p_miss, 3),
    }
