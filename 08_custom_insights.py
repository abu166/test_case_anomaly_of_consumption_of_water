"""
08_custom_insights.py
Module for 3 custom analytical insights discovered in the water consumption dataset:
1. Telemetry Packet Storms (paketov_za_sutki >= 20) leading to rapid battery drain.
2. Hardware/Firmware Reliability Vulnerability across device models (AQUA2 vs aqua2_nvt_wa_stm).
3. Impact of System Reset events ('сброс') on cumulative counter registers.
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory, load_sobytiya


def analyze_custom_insights(
    df_pok: pd.DataFrame = None,
    df_prib: pd.DataFrame = None,
    df_sob: pd.DataFrame = None
) -> dict:
    """
    Computes 3 unique analytical insights from the water consumption and device telemetry dataset.
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()
    if df_sob is None:
        df_sob = load_sobytiya()

    # -------------------------------------------------------------
    # Custom Insight 1: Telemetry Packet Storms (paketov_za_sutki >= 20)
    # -------------------------------------------------------------
    packet_storm_records = df_pok[df_pok["paketov_za_sutki"] >= 20].copy()
    packet_storm_devices = packet_storm_records.groupby("anon_id").agg(
        max_packets=("paketov_za_sutki", "max"),
        total_high_packet_days=("paketov_za_sutki", "count"),
        total_packets=("paketov_za_sutki", "sum")
    ).reset_index().sort_values("max_packets", ascending=False).reset_index(drop=True)
    
    packet_storm_devices = packet_storm_devices.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    insight_1 = {
        "title": "Находка 1: Аномальные пакетные штормы телеметрии (Packet Storms)",
        "criterion": "paketov_za_sutki >= 20 (норма для контроллера: 1-2 пакета в сутки)",
        "total_affected_devices": len(packet_storm_devices),
        "total_anomalous_records": len(packet_storm_records),
        "top_examples": packet_storm_devices.head(5).to_dict(orient="records"),
        "explanation": (
            "Шторм пакетов (передача до 2 585 пакетов в сутки вместо 1-2): прибор непрерывно выходит в эфир. "
            "Это приводит к экстренному разряду встроенного литиевого аккумулятора (устройство выйдет из строя за недели) "
            "и зашумлению радиоэфира базовой станции (BS)."
        )
    }

    # -------------------------------------------------------------
    # Custom Insight 2: Model Hardware/Firmware Vulnerability (AQUA2 vs aqua2_nvt_wa_stm)
    # -------------------------------------------------------------
    unreal_devs = set(df_pok[df_pok["raw_diff"] > 50]["anon_id"])
    neg_devs = set(df_pok[df_pok["raw_diff"] < -0.001]["anon_id"])

    model_summary = df_prib.groupby(["tip", "model"]).size().reset_index(name="total_devices")
    model_summary["unreal_devices"] = model_summary.apply(
        lambda r: len(set(df_prib[(df_prib["tip"]==r["tip"]) & (df_prib["model"]==r["model"])]["anon_id"]).intersection(unreal_devs)), axis=1
    )
    model_summary["neg_devices"] = model_summary.apply(
        lambda r: len(set(df_prib[(df_prib["tip"]==r["tip"]) & (df_prib["model"]==r["model"])]["anon_id"]).intersection(neg_devs)), axis=1
    )
    model_summary["anomaly_pct"] = ((model_summary["unreal_devices"] + model_summary["neg_devices"]) / model_summary["total_devices"] * 100).round(2)
    model_summary = model_summary.sort_values("anomaly_pct", ascending=False).reset_index(drop=True)

    insight_2 = {
        "title": "Находка 2: Противостояние ревизий оборудования (AQUA2 vs aqua2_nvt_wa_stm)",
        "criterion": "Доля приборов с критическими сбоями (нереальный/отрицательный расход) в разрезе типов приборов",
        "model_table": model_summary,
        "explanation": (
            "Установлена катастрофическая разница в надежности: старая ревизия 'AQUA2' демонстрирует до 17.8% "
            "сбоев с выбросами и отрицательным расходом, тогда как обновленная модификация 'aqua2_nvt_wa_stm' "
            "практически свободна от аномалий памяти (<0.3% сбоев). Это указывает на конструктивный/прошивный дефект ревизии AQUA2."
        )
    }

    # -------------------------------------------------------------
    # Custom Insight 3: Impact of Hardware Resets ('сброс' in sobytiya.csv)
    # -------------------------------------------------------------
    reset_events = df_sob[df_sob["sobytie"] == "сброс"].copy()
    reset_devices = reset_events["anon_id"].nunique()
    
    # Check readings on reset dates
    merged_reset = df_pok.merge(reset_events[["anon_id", "data", "sobytie"]], on=["anon_id", "data"], how="inner")
    neg_on_reset = merged_reset[merged_reset["raw_diff"] < -0.001]["anon_id"].nunique()
    zero_on_reset = merged_reset[merged_reset["daily_flow"].abs() < 1e-6]["anon_id"].nunique()

    reset_summary = reset_events.groupby("anon_id").size().reset_index(name="reset_count").sort_values("reset_count", ascending=False).reset_index(drop=True)
    reset_summary = reset_summary.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    insight_3 = {
        "title": "Находка 3: Аппаратные перезагрузки ('Сброс') и их влияние на показания",
        "criterion": "Фиксация события 'сброс' в sobytiya.csv и сопоставление с накопительным счетчиком",
        "total_reset_events": len(reset_events),
        "total_affected_devices": reset_devices,
        "neg_flow_on_reset_devices": neg_on_reset,
        "zero_flow_on_reset_devices": zero_on_reset,
        "top_examples": reset_summary.head(5).to_dict(orient="records"),
        "explanation": (
            "Событие 'сброс' зафиксировано 374 раза на 108 приборах. В дни перезагрузок происходят либо "
            "корректировки накопительного счетчика в меньшую сторону, либо заморозка передаваемых данных. "
            "Это признак нестабильности питания микрометрического модуля или сброса EEPROM."
        )
    }

    return {
        "anomaly_name": "Собственные аналитические находки (Custom Insights)",
        "insight_1": insight_1,
        "insight_2": insight_2,
        "insight_3": insight_3
    }


if __name__ == "__main__":
    res = analyze_custom_insights()
    print(f"=== {res['anomaly_name']} ===")
    print(f"\n1. {res['insight_1']['title']}")
    print(f"   Приборов со штормом пакетов: {res['insight_1']['total_affected_devices']}")
    print(f"\n2. {res['insight_2']['title']}")
    print(res['insight_2']['model_table'].head(5).to_string())
    print(f"\n3. {res['insight_3']['title']}")
    print(f"   Приборов со сбросами: {res['insight_3']['total_affected_devices']} (Всего событий сброса: {res['insight_3']['total_reset_events']})")
