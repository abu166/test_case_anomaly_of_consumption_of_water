"""
01_negative_flow.py
Module for detecting Negative Flow Anomaly (Meter going backwards / cumulative reading decreased).
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory


def analyze_negative_flow(df_pok: pd.DataFrame = None, df_prib: pd.DataFrame = None) -> dict:
    """
    Identifies devices where meter reading decreased (pokazanie < prev_pokazanie).
    
    Criterion: raw_diff < -0.001 m³ (to exclude minor numerical noise).
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()

    # Filter negative flow records
    neg_records = df_pok[df_pok["raw_diff"] < -0.001].copy()
    
    # Device-level aggregation
    device_summary = neg_records.groupby("anon_id").agg(
        negative_events_count=("raw_diff", "count"),
        min_negative_diff=("raw_diff", "min"),
        total_negative_volume=("raw_diff", "sum"),
        first_negative_date=("data", "min")
    ).reset_index()

    # Sort by magnitude of negative drop
    device_summary = device_summary.sort_values("min_negative_diff", ascending=True).reset_index(drop=True)
    
    # Merge device metadata
    device_summary = device_summary.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")
    
    # Selected top example devices
    top_examples = device_summary.head(5).to_dict(orient="records")

    result = {
        "anomaly_name": "Отрицательный расход (Negative Flow)",
        "criterion": "raw_diff < -0.001 m³ (накопительное показание уменьшилось по сравнению с предыдущим днем)",
        "total_anomalous_records": int(len(neg_records)),
        "total_affected_devices": int(len(device_summary)),
        "affected_device_ids": device_summary["anon_id"].tolist(),
        "device_summary": device_summary,
        "top_examples": top_examples,
        "causes_explanation": (
            "Основные причины отрицательного расхода:\n"
            "1. Коррупция пакета телеметрии (переполнение 32-битного целочисленного значения/битовые сбои), "
            "когда прибор временно передает заведомо ложное мусорное значение, а затем возвращается к норме.\n"
            "2. Физическая замена счетчика или сброс показаний ('сброс') без фиксации нового стартового значения.\n"
            "3. Обратный ток воды (backflow) при отсутствии обратного клапана и перепадах давления в стояке."
        )
    }
    return result


if __name__ == "__main__":
    res = analyze_negative_flow()
    print(f"=== {res['anomaly_name']} ===")
    print(f"Критерий: {res['criterion']}")
    print(f"Количество найденных приборов: {res['total_affected_devices']}")
    print(f"Всего записей с отрицательным расходом: {res['total_anomalous_records']}")
    print("\nПримеры приборов:")
    for ex in res["top_examples"]:
        print(f"  - anon_id: {ex['anon_id']} | Мин. сброс: {ex['min_negative_diff']:.3f} м³ | Тип: {ex['tip']} ({ex['model']})")
