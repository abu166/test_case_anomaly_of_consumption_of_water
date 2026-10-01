"""
03_unrealistic_flow.py
Module for detecting Unrealistic Values Anomaly (Physically impossible daily flow: tens/thousands/millions of m³).
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory


def analyze_unrealistic_flow(
    df_pok: pd.DataFrame = None,
    df_prib: pd.DataFrame = None,
    unrealistic_threshold: float = 50.0
) -> dict:
    """
    Identifies devices with physically impossible daily water flow (daily_flow > 50 m³/day).
    
    Standard 15mm domestic water pipe (D15) has maximum flow rate ~1.5 - 2.5 m³/hour (~36-60 m³/day full capacity).
    Flows > 50 m³/day (and up to millions of m³/day) represent bit overflow / garbage payload corrupted values.
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()

    # Filter records with daily flow > 50 m³
    unreal_records = df_pok[df_pok["raw_diff"] > unrealistic_threshold].copy()

    # Aggregate by device
    device_summary = unreal_records.groupby("anon_id").agg(
        unrealistic_events_count=("raw_diff", "count"),
        max_unrealistic_flow=("raw_diff", "max"),
        total_unrealistic_volume=("raw_diff", "sum"),
        first_unrealistic_date=("data", "min")
    ).reset_index()

    device_summary = device_summary.sort_values("max_unrealistic_flow", ascending=False).reset_index(drop=True)
    device_summary = device_summary.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    top_examples = device_summary.head(5).to_dict(orient="records")

    result = {
        "anomaly_name": "Нереальные значения расхода (Unrealistic Values)",
        "criterion": f"raw_diff > {unrealistic_threshold} м³/сут (физически невозможный расход для бытового прибора учета)",
        "threshold": unrealistic_threshold,
        "total_anomalous_records": int(len(unreal_records)),
        "total_affected_devices": int(len(device_summary)),
        "affected_device_ids": device_summary["anon_id"].tolist(),
        "device_summary": device_summary,
        "top_examples": top_examples,
        "causes_explanation": (
            "Основные причины нереальных значений:\n"
            "1. Битовые сбои в памяти прибора учета или контроллера телеметрии (запись сбойного инта в регистр).\n"
            "2. Переполнение разрядной сетки телеметрического счетчика (integer rollover).\n"
            "3. Коррупция пакета данных при беспроводной передаче (отсутствие CRC проверки на уровне прошивки)."
        )
    }
    return result


if __name__ == "__main__":
    res = analyze_unrealistic_flow()
    print(f"=== {res['anomaly_name']} ===")
    print(f"Критерий: {res['criterion']}")
    print(f"Количество найденных приборов: {res['total_affected_devices']}")
    print(f"Всего нереальных записей: {res['total_anomalous_records']}")
    print("\nПримеры приборов:")
    for ex in res["top_examples"]:
        print(f"  - anon_id: {ex['anon_id']} | Макс. суточный расход: {ex['max_unrealistic_flow']:,.2f} м³ | {ex['tip']} ({ex['model']})")
