"""Merkezi yol ve ayar tanimlari.

Butun script'ler dosya yollarini buradan alir; boylece proje kokunden mi
yoksa alt klasorden mi calistirildigi fark etmez.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
WEB_DIR = ROOT / "web"

TRAINING_PAIRS_PATH = DATA_DIR / "training_pairs_2024_07.parquet"
MODEL_PATHS = {
    0.1: MODELS_DIR / "delay_model_q10.txt",
    0.5: MODELS_DIR / "delay_model_q50.txt",
    0.9: MODELS_DIR / "delay_model_q90.txt",
}

# --- DB Timetables API ---
DB_CLIENT_ID = os.environ.get("DB_CLIENT_ID")
DB_API_KEY = os.environ.get("DB_API_KEY")
DB_BASE_URL = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
DB_HEADERS = {
    "DB-Client-Id": DB_CLIENT_ID or "",
    "DB-Api-Key": DB_API_KEY or "",
    "Accept": "application/xml",
}

# --- Kaynak veri seti ---
HF_REPO_ID = "piebro/deutsche-bahn-data"
HF_SAMPLE_FILE = "monthly_processed_data/data-2024-07.parquet"

# --- Izleme / risk parametreleri ---
MONITOR_EVA = "8000105"      # Frankfurt(Main)Hbf
BUFFER_TIME_MIN = 9.0        # varsayimsal aktarma tampon suresi
STATIONS_BETWEEN = 1
RISK_THRESHOLD = 0.4


def require_credentials() -> None:
    """API anahtarlari yoksa net bir hata ver."""
    if not DB_CLIENT_ID or not DB_API_KEY:
        raise SystemExit(
            "DB_CLIENT_ID / DB_API_KEY bulunamadi.\n"
            f"'.env.example' dosyasini '{ROOT / '.env'}' olarak kopyalayip doldur."
        )


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
