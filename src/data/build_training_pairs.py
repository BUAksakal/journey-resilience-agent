"""Egitim ciftlerini olusturur: "erken duraktaki gecikme -> gec duraktaki gecikme".

Calistirma (proje kokunden):
    python -m src.data.build_training_pairs
"""

import pandas as pd
from huggingface_hub import hf_hub_download

from src import config


def load_raw_data() -> pd.DataFrame:
    local_path = hf_hub_download(
        repo_id=config.HF_REPO_ID, repo_type="dataset", filename=config.HF_SAMPLE_FILE
    )
    return pd.read_parquet(local_path)


def build_before_after_pairs(df, min_gap_stations=1, max_gap_stations=3):
    """
    ONEMLI: train_line_ride_id tek basina bir GUNU degil, tekrar eden bir
    seferi temsil eder -- ayni id farkli haftalarda tekrar goruluyor.
    Tek gunluk yolculuk kimligi 'id' alanindan turetilir:
        id               = "{ride_id}-{sefer_zaman_damgasi}-{durak_no}"
        trip_instance_id = "{ride_id}-{sefer_zaman_damgasi}"
    """
    df = df.dropna(subset=["id", "train_line_station_num", "delay_in_min"]).copy()
    df["trip_instance_id"] = df["id"].str.rsplit("-", n=1).str[0]
    df = df.sort_values(["trip_instance_id", "train_line_station_num"]).reset_index(drop=True)
    grouped = df.groupby("trip_instance_id", sort=False)

    all_pairs = []
    for gap in range(min_gap_stations, max_gap_stations + 1):
        late_delay = grouped["delay_in_min"].shift(-gap)
        late_station = grouped["station_name"].shift(-gap)
        valid = late_delay.notna()

        chunk = pd.DataFrame({
            "trip_instance_id": df.loc[valid, "trip_instance_id"].values,
            "train_type": df.loc[valid, "train_type"].values,
            "early_station": df.loc[valid, "station_name"].values,
            "late_station": late_station[valid].values,
            "stations_between": gap,
            "early_delay_min": df.loc[valid, "delay_in_min"].values,
            "late_delay_min": late_delay[valid].values,
        })
        chunk["delay_change_min"] = chunk["late_delay_min"] - chunk["early_delay_min"]
        all_pairs.append(chunk)
        print(f"  gap={gap}: {len(chunk)} cift uretildi")

    return pd.concat(all_pairs, ignore_index=True)


def main():
    config.ensure_dirs()
    print("Veri yukleniyor...")
    df = load_raw_data()
    print(f"{len(df)} satir yuklendi.")

    print("\nOnce-sonra ciftleri olusturuluyor...")
    pairs = build_before_after_pairs(df)
    print(f"{len(pairs)} egitim cifti olusturuldu.")

    print("\nOrnek satirlar:")
    print(pairs.head(5).to_string())
    print("\nTemel istatistik (delay_change_min):")
    print(pairs["delay_change_min"].describe())

    pairs.to_parquet(config.TRAINING_PAIRS_PATH)
    print(f"\nKaydedildi -> {config.TRAINING_PAIRS_PATH}")


if __name__ == "__main__":
    main()
