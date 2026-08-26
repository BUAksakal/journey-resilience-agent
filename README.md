# Journey Resilience Agent

A live transfer-risk monitor for German rail journeys. Instead of telling you a connection
failed, it estimates — continuously, while you're travelling — how likely you are to miss it,
and branches to replanning before the connection closes.

Built with **LangGraph** (orchestration), **LightGBM quantile regression** (risk model),
**FastAPI** (serving) and the official **DB Timetables API** (live data).

---

## What it actually does

```
monitor ──▶ assess_risk ──▶ [decide] ──┬──▶ replan ──▶ notify
   ▲                                    └──▶ wait   ──▶ notify
   │
DB Timetables API (live delay + cause code)
```

1. **monitor** — reads a station's live board (`/fchg`): current delay (`ct − pt`),
   platform change, and the official delay-cause code.
2. **assess_risk** — a LightGBM quantile model predicts where that delay lands by arrival,
   as a calibrated range, then converts it to a missed-transfer probability against the
   scheduled buffer.
3. **decide** — a conditional edge: above the risk threshold the graph branches to replanning,
   below it, monitoring continues.
4. **notify** — surfaces the result. In production this is where an LLM turns the numbers into
   plain language; everything upstream is deterministic.

---

## Quick start

```bash
git clone https://github.com/<user>/journey-resilience-agent.git
cd journey-resilience-agent

python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your DB API credentials
```

Get free credentials at [developers.deutschebahn.com](https://developers.deutschebahn.com):
create an application, subscribe it to the **Timetables** product (free tier, instant approval),
and copy the Client ID and Client Secret into `.env`.

### 1. Verify API access

```bash
python -m scripts.check_apis
```

### 2. Build the training data and train the model

```bash
python -m src.data.build_training_pairs    # ~5.6M before/after delay pairs
python -m src.ml.train_delay_model         # writes models/delay_model_q{10,50,90}.txt
```

### 3. Run the agent once against live data

```bash
python -m src.agent.graph
```

### 4. Run the API + web interface

```bash
uvicorn src.api.server:app --reload --port 8000
```

Then open `web/index.html` in a browser. The header shows a live/offline indicator for the backend.

---

## Project structure

```
journey-resilience-agent/
├── src/
│   ├── config.py                    # all paths, credentials, thresholds
│   ├── db_client.py                 # DB Timetables API wrapper
│   ├── data/build_training_pairs.py # historical data → training pairs
│   ├── ml/train_delay_model.py      # LightGBM quantile training
│   ├── ml/risk.py                   # inference + missed-transfer probability
│   ├── agent/graph.py               # LangGraph state machine
│   └── api/server.py                # FastAPI service
├── scripts/check_apis.py            # data-source connectivity diagnostics
├── web/index.html                   # landing page + live query interface
├── models/                          # trained models (generated)
├── data/                            # training pairs (generated, gitignored)
└── .env                             # credentials (never committed)
```

---

## Model performance

Trained on 1.87M cleaned before/after delay pairs (July 2024), split by journey so no
single trip appears in both train and test.

| Quantile | Target coverage | Actual coverage |
|---|---|---|
| q50 | 50% | 56.2% |
| q90 | 90% | 90.7% |
| q10 | 10% | 29.0% |

**q90 — the quantile the risk decision depends on — is well calibrated.** q10 is not, and this
is disclosed rather than hidden: delay values cluster heavily at 0–2 minutes, which makes the
lower tail hard to separate. Because the safety-relevant decision is made against the pessimistic
end of the range, the miscalibrated optimistic bound does not drive the outcome. Fixing it is
tracked as known work, not quietly ignored.

A second known limitation: gradient-boosted trees do not extrapolate. A delay far outside the
training range collapses to the nearest learned value rather than extending sensibly — which is
precisely why calibration monitoring is part of the design rather than an afterthought.

---

## Scope and honest limitations

- **No automatic route search.** The only accessible official API (Timetables) is single-station:
  it answers "what is happening at this station", never "how do I get from A to B". The entire
  RIS family (Connections, Journeys, Boards, Stations, Disruptions) requires vetting as an approved
  DB sales partner with contract-negotiated pricing, and the community wrapper `v6.db.transport.rest`
  is blocked at TLS level from the development network. Journeys are therefore defined by leg, and
  replanning selects from a predefined candidate list. This is a disclosed scope decision.
- **No platform-level transfer times.** `RIS::Stations` is the only source for real platform-to-platform
  walking times and is inaccessible, so the risk calculation uses scheduled buffer minus predicted
  delay, with no guessed minimum-walk constant.
- **Live window only.** The Timetables API covers roughly ±18 hours, so historical training data comes
  from a third-party archive rather than DB directly.

---

## Data sources and attribution

- **Live:** [DB Timetables API](https://developers.deutschebahn.com), official, free tier.
- **Historical:** [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data),
  CC BY 4.0.
- **Delay-cause codes:** decoded via the community-maintained
  [Travel::Status::DE::IRIS](https://github.com/derf/Travel-Status-DE-IRIS) mapping, cross-checked
  against DB's own IRIS webclient configuration. DB does not publish this table itself, so treat it
  as unofficial-but-verified.
- **Prior art:** [Bahnvorhersage](https://bahnvorhersage.de) demonstrated ML-based transfer-success
  prediction (open source, Prototype Fund, presented at 38C3), and
  [Trainator](https://trainator.eu) offers probabilistic delay distributions commercially. This
  project does not claim to have invented delay prediction — the contribution is the live
  orchestration layer that acts on a probability during an active journey.

## License

MIT
