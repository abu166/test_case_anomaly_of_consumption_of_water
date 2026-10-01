"""
02_spikes.py
Module for detecting Spikes Anomaly (Anomalously high daily consumption).
Requires BOTH an absolute threshold and a relative threshold.
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory


def analyze_spikes(
    df_pok: pd.DataFrame = None,
    df_prib: pd.DataFrame = None,
    abs_min_threshold: float = 3.0,
    abs_max_threshold: float = 50.0,
    relative_multiplier: float = 5.0
) -> dict:
    """
    Identifies consumption spikes using dual criteria:
    - Absolute threshold: abs_min_threshold < daily_flow <= abs_max_threshold (e.g. 3.0 to 50.0 m³/day)
    - Relative threshold: daily_flow >= relative_multiplier * median_flow (e.g. 5x normal flow)
    
    (Values > abs_max_threshold are classified under Unrealistic Values).
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()

    # Calculate median daily consumption per device on non-zero days
    valid_daily = df_pok[(df_pok["date_diff"] == 1) & (df_pok["daily_flow"] > 0.001)].copy()
    device_medians = valid_daily.groupby("anon_id")["daily_flow"].median().rename("median_daily_flow").reset_index()

    # Merge median flow back to daily records
    merged = df_pok.merge(device_medians, on="anon_id", how="left")
    # For devices with no positive flow days, fill median with 0.1 m³
    merged["median_daily_flow"] = merged["median_daily_flow"].fillna(0.1)

    # Spike condition
    spike_mask = (
        (merged["daily_flow"] > abs_min_threshold) &
        (merged["daily_flow"] <= abs_max_threshold) &
        (merged["daily_flow"] >= relative_multiplier * merged["median_daily_flow"])
    )

    spike_records = merged[spike_mask].copy()

    # Aggregate by device
    device_summary = spike_records.groupby("anon_id").agg(
        spike_count=("daily_flow", "count"),
        max_spike_flow=("daily_flow", "max"),
        median_daily_flow=("median_daily_flow", "first"),
        first_spike_date=("data", "min")
    ).reset_index()

    device_summary["max_spike_ratio"] = device_summary["max_spike_flow"] / device_summary["median_daily_flow"]
    device_summary = device_summary.sort_values("max_spike_flow", ascending=False).reset_index(drop=True)
    device_summary = device_summary.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    top_examples = device_summary.head(5).to_dict(orient="records")

    result = {
        "anomaly_name": "Всплески потребления (Consumption Spikes)",
        "criterion": (
            f"Двойной порог: Абсолютный ({abs_min_threshold} м³/сут < daily_flow <= {abs_max_threshold} м³/сут) "
            f"И Относительный (daily_flow >= {relative_multiplier}x от медианного суточного расхода прибора)"
        ),
        "abs_min_threshold": abs_min_threshold,
        "abs_max_threshold": abs_max_threshold,
        "relative_multiplier": relative_multiplier,
        "total_anomalous_records": int(len(spike_records)),
        "total_affected_devices": int(len(device_summary)),
        "affected_device_ids": device_summary["anon_id"].tolist(),
        "device_summary": device_summary,
        "top_examples": top_examples,
        "causes_explanation": (
            "Основные причины всплесков потребления:\n"
            "1. Разовый залповый расход (набор бассейна, интенсивные ремонтные/клининговые работы).\n"
            "2. Временная тяжелая авария/поломка арматуры (прорыв шланга, заклинивший клапан сливного бачка на несколько часов).\n"
            "3. Корректирующий пропуск телеметрии: когда телеметрия отсутствовала, а при поступлении следующего пакета разница записалась в один день."
        )
    }
    return result


if __name__ == "__main__":
    res = analyze_spikes()
    print(f"=== {res['anomaly_name']} ===")
    print(f"Критерий: {res['criterion']}")
    print(f"Количество найденных приборов: {res['total_affected_devices']}")
    print(f"Всего аномальных всплесков: {res['total_anomalous_records']}")
    print("\nПримеры приборов:")
    for ex in res["top_examples"]:
        print(f"  - anon_id: {ex['anon_id']} | Макс. всплеск: {ex['max_spike_flow']:.2f} м³/сут (Медиана: {ex['median_daily_flow']:.3f} м³, Превышение: {ex['max_spike_ratio']:.1f}x) | {ex['tip']}")
