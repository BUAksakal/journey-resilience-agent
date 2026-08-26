"""Kullanilan/denenen veri kaynaklarina erisimi test eder.

Calistirma (proje kokunden):
    python -m scripts.check_apis
"""

import requests

from src import config, db_client

ENDPOINTS = {
    "DB Timetables API (kullaniliyor)": ("station", "Frankfurt Hbf"),
}


def main():
    config.require_credentials()
    print("=" * 60)
    print("DB Timetables API testi")
    print("=" * 60)
    for name in ("Frankfurt Hbf", "Mannheim Hbf", "Stuttgart Hbf"):
        try:
            stations = db_client.find_station(name)
            for s in stations:
                print(f"  {s.get('name'):<22} EVA={s.get('eva')}  DS100={s.get('ds100')}")
        except Exception as e:
            print(f"  {name}: HATA {e}")

    print("\nCanli gecikme akisi ornegi:")
    info = db_client.first_delayed_train(config.MONITOR_EVA)
    print(" ", info if info else "su an gecikme mesaji tasiyan tren yok")

    print("\n" + "=" * 60)
    print("Erisilemedigi dogrulanan kaynaklar (referans)")
    print("=" * 60)
    for url, note in [
        ("https://v6.db.transport.rest/locations?query=Berlin+Hbf",
         "topluluk sarmalayicisi -- gelistirme aginda TLS seviyesinde engelli"),
        ("https://bahnvorhersage.de", "referans/benchmark, entegre edilmiyor"),
    ]:
        try:
            r = requests.get(url, timeout=8)
            print(f"  OK ({r.status_code}) {url}  # {note}")
        except Exception as e:
            print(f"  ERISILEMEDI {url}  # {note} -- {type(e).__name__}")


if __name__ == "__main__":
    main()
