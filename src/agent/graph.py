"""LangGraph durum makinesi: monitor -> assess_risk -> decide -> replan/wait -> notify.

Calistirma (proje kokunden):
    python -m src.agent.graph
"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from src import config, db_client
from src.ml import risk


class JourneyState(TypedDict):
    current_delay_min: Optional[float]
    cause_code: Optional[int]
    train_type: Optional[str]
    train_number: Optional[str]
    q10: Optional[float]
    q50: Optional[float]
    q90: Optional[float]
    p_miss: Optional[float]
    message: Optional[str]


def monitor_node(state: JourneyState) -> JourneyState:
    print(f"[monitor] EVA {config.MONITOR_EVA} canli veri cekiliyor...")
    info = db_client.first_delayed_train(config.MONITOR_EVA)
    if info is None:
        print("[monitor] Su an aktif gecikme mesaji tasiyan tren yok.")
        return {**state, "current_delay_min": None}
    print(f"[monitor] {info['train_type']} {info['train_number']} | "
          f"gecikme={info['delay_min']:.0f} dk | neden_kodu={info['cause_code']}")
    return {**state, **{
        "current_delay_min": info["delay_min"],
        "cause_code": info["cause_code"],
        "train_type": info["train_type"],
        "train_number": info["train_number"],
    }}


def assess_risk_node(state: JourneyState) -> JourneyState:
    if state.get("current_delay_min") is None:
        return {**state, "p_miss": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}
    result = risk.assess(state["current_delay_min"], state["train_type"])
    print(f"[assess_risk] tahmini varis gecikmesi -> "
          f"q10={result['q10']} q50={result['q50']} q90={result['q90']}")
    print(f"[assess_risk] tampon {config.BUFFER_TIME_MIN} dk -> p_miss={result['p_miss']:.2%}")
    return {**state, **result}


def decide(state: JourneyState) -> str:
    p = state.get("p_miss") or 0.0
    branch = "replan" if p >= config.RISK_THRESHOLD else "wait"
    print(f"[decide] p_miss={p:.2%} vs esik={config.RISK_THRESHOLD:.0%} -> {branch.upper()}")
    return branch


def replan_node(state: JourneyState) -> JourneyState:
    # TODO: onceden tanimli yedek baglanti listesini kontrol et (CLAUDE.md SS7.3).
    return {**state, "message": (
        f"Aktarmayi kacirma riski yuksek (%{(state['p_miss'] or 0) * 100:.0f}). "
        "Yedek baglanti kontrolu henuz baglanmadi."
    )}


def wait_node(state: JourneyState) -> JourneyState:
    return {**state, "message": (
        f"Aktarma guvende gorunuyor (risk %{(state.get('p_miss') or 0) * 100:.0f}). Izleme suruyor."
    )}


def notify_node(state: JourneyState) -> JourneyState:
    # Gercek sistemde bu metni bir LLM dogal dile cevirir (CLAUDE.md SS7.3).
    print(f"\n[notify] {state['message']}\n")
    return state


def build_graph():
    g = StateGraph(JourneyState)
    g.add_node("monitor", monitor_node)
    g.add_node("assess_risk", assess_risk_node)
    g.add_node("replan", replan_node)
    g.add_node("wait", wait_node)
    g.add_node("notify", notify_node)

    g.set_entry_point("monitor")
    g.add_edge("monitor", "assess_risk")
    g.add_conditional_edges("assess_risk", decide, {"replan": "replan", "wait": "wait"})
    g.add_edge("replan", "notify")
    g.add_edge("wait", "notify")
    g.add_edge("notify", END)
    return g.compile()


def main():
    config.require_credentials()
    app = build_graph()
    empty: JourneyState = dict.fromkeys(JourneyState.__annotations__, None)  # type: ignore
    final = app.invoke(empty)
    print("=" * 60)
    print("Nihai durum:", final)


if __name__ == "__main__":
    main()
