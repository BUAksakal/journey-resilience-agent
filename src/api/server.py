"""Web arayuzunun cagirdigi FastAPI servisi.

Calistirma (proje kokunden):
    uvicorn src.api.server:app --reload --port 8000
"""

import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import config, db_client
from src.ml import risk

app = FastAPI(title="Journey Resilience Agent API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    config.require_credentials()


@app.get("/api/health")
def health():
    return {"status": "ok", "monitoring_eva": config.MONITOR_EVA}


@app.get("/api/check")
def check_train(train_number: str):
    digits = re.sub(r"\D", "", train_number)
    if not digits:
        raise HTTPException(400, "Gecerli bir tren numarasi gir (orn. 596).")

    info = db_client.find_train(config.MONITOR_EVA, digits)
    if info is None:
        raise HTTPException(
            404,
            f"Tren {digits} su an bu istasyonun canli akisinda bulunamadi "
            "(kapsam: sadece izlenen istasyondan gecen trenler).",
        )

    result = risk.assess(info["delay_min"], info["train_type"] or "ICE")
    return {
        "train_number": digits,
        "train_type": info["train_type"],
        "current_delay_min": round(info["delay_min"], 1),
        "cause_code": info["cause_code"],
        "predicted_delay_q10": result["q10"],
        "predicted_delay_q50": result["q50"],
        "predicted_delay_q90": result["q90"],
        "p_miss": result["p_miss"],
    }
