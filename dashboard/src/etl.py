#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etlcontugas.py
ETL de dataset Contugas:
- Lee un Excel .xlsx con múltiples hojas (por defecto: dataset_contugas.xlsx).
- Añade columna 'Cliente' con el nombre de la hoja.
- Normaliza y parsea la columna 'Fecha' (múltiples formatos).
- Detecta y elimina duplicados por ['Cliente','Fecha'].
- Imputa 'Presion' cuando Presion==0 y Volumen>0 usando el promedio de las 3 horas previas por cliente.
- Verifica y setea Volumen y Presion negativos a cero. Temperatura puede tener valores negativos.
- Segmenta clientes en 'Segmento' (Comercial/Industrial) usando DBSCAN con parámetros fijos (eps=0.3425, min_samples=3) y lógica basada en Volumen.
- Exporta df_contugas.csv (UTF-8-SIG) y un reporte de imputación imputacion_presion_report.csv.

Uso:
    python etlcontugas.py
    python etlcontugas.py --input otra_ruta.xlsx
    python etlcontugas.py --output df_contugas.csv
    python etlcontugas.py --input otra.xlsx --output df_contugas.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# --- Formatos conocidos de la columna Fecha ---
KNOWN_FORMATS: List[str] = [
    "%d-%m-%Y %I:%M:%S %p",
    "%d-%m-%Y %I:%M %p",
    "%d-%m-%y %H:%M",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
]

DEFAULT_INPUT = "dataset_contugas.xlsx"
DEFAULT_OUTPUT = "df_contugas.csv"


def parse_dates_series(s: pd.Series) -> pd.Series:
    """
    Parsea 'Fecha' manejando mezclas de formatos:
    - ISO (YYYY-MM-DD [HH:MM[:SS]]) -> dayfirst=False
    - El resto (e.g., DD-MM-YYYY, DD/MM/YYYY) -> dayfirst=True
    - Reintenta con formatos explícitos si quedan NaT
    """
    s_norm = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Resultado inicial
    dt = pd.Series(pd.NaT, index=s_norm.index, dtype="datetime64[ns]")

    # 1) ISO primero (evita el warning de dayfirst con YYYY-MM-DD)
    iso_mask = s_norm.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
    if iso_mask.any():
        dt.loc[iso_mask] = pd.to_datetime(
            s_norm[iso_mask],
            errors="coerce",
            dayfirst=False,
            utc=False,
        )

    # 2) No-ISO con dayfirst=True (DD-MM-YYYY, DD/MM/YYYY, etc.)
    other_mask = ~iso_mask
    if other_mask.any():
        dt.loc[other_mask] = pd.to_datetime(
            s_norm[other_mask],
            errors="coerce",
            dayfirst=True,
            utc=False,
        )

    # 3) Reintento con formatos explícitos para lo que quede NaT
    mask_nat = dt.isna()
    if mask_nat.any():
        for fmt in KNOWN_FORMATS:
            trial = pd.to_datetime(s_norm[mask_nat], format=fmt, errors="coerce")
            dt.loc[mask_nat] = dt.loc[mask_nat].fillna(trial)
            mask_nat = dt.isna()
            if not mask_nat.any():
                break

    return dt


def load_excel_with_client_column(xlsx_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    xls = pd.ExcelFile(xlsx_path)
    per_sheet_counts: Dict[str, int] = {}
    frames: List[pd.DataFrame] = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df["Cliente"] = sheet_name
        per_sheet_counts[sheet_name] = len(df)
        frames.append(df)
    if not frames:
        raise ValueError("El archivo Excel no contiene hojas legibles.")
    df_all = pd.concat(frames, ignore_index=True)
    return df_all, per_sheet_counts


def validate_totals(per_sheet_counts: Dict[str, int], df_concat: pd.DataFrame) -> bool:
    sum_sheets = sum(per_sheet_counts.values())
    total_concat = len(df_concat)
    ok = (sum_sheets == total_concat)
    print("=== Validación de totales (antes de deduplicar) ===")
    for k, v in per_sheet_counts.items():
        print(f"  - {k}: {v} registros")
    print(f"  Suma por hojas: {sum_sheets}")
    print(f"  Total concatenado: {total_concat}")
    print(f"  Resultado: {'OK' if ok else 'NO COINCIDE'}")
    return ok


def ensure_fecha_column(df: pd.DataFrame) -> None:
    if "Fecha" in df.columns:
        return
    candidates = [c for c in df.columns if c.strip().lower() in {"fecha", "date", "datetime", "fecha_hora", "fechahora"}]
    if candidates:
        df.rename(columns={candidates[0]: "Fecha"}, inplace=True)
        print(f"[INFO] Renombrada columna '{candidates[0]}' -> 'Fecha'")
        return
    raise KeyError("No se encontró la columna 'Fecha'.")


def drop_duplicates_by_cliente_fecha(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df_sorted = df.sort_values(["Cliente", "Fecha"]).copy()
    df_dedup = df_sorted.drop_duplicates(subset=["Cliente", "Fecha"], keep="first")
    removed = before - len(df_dedup)
    return df_dedup, removed


def imputar_presion_ceros_con_media_prev_3h(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["Volumen"] = pd.to_numeric(df.get("Volumen"), errors="coerce")
    df["Presion"] = pd.to_numeric(df.get("Presion"), errors="coerce")
    df = df.dropna(subset=["Fecha", "Cliente", "Volumen", "Presion"])
    df = df.sort_values(["Cliente", "Fecha"]).reset_index(drop=True)

    df["__mean_prev3h"] = np.nan
    for cli, g in df.groupby("Cliente", sort=False):
        g = g.sort_values("Fecha")
        s = g["Presion"].copy()
        s.index = g["Fecha"]
        m = s.rolling("3h", closed="left").mean()
        df.loc[g.index, "__mean_prev3h"] = m.values

    mask_candidates = (df["Presion"] == 0) & (df["Volumen"] > 0)
    mask_imputable = mask_candidates & df["__mean_prev3h"].notna()
    df.loc[mask_imputable, "Presion"] = df.loc[mask_imputable, "__mean_prev3h"]

    rep = (
        df.loc[mask_imputable, ["Cliente", "__mean_prev3h"]]
          .groupby("Cliente")
          .agg(registros_imputados=("__mean_prev3h", "size"),
               presion_imputada_promedio=("__mean_prev3h", "mean"))
          .reset_index()
    )
    rep["presion_imputada_promedio"] = rep["presion_imputada_promedio"].astype(float)

    df.drop(columns=["__mean_prev3h"], inplace=True)
    return df, rep


# === Función para asegurar no-negatividad en Volumen y Presion ===
def corregir_valores_negativos(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    if "Volumen" in df.columns:
        df["Volumen"] = pd.to_numeric(df["Volumen"], errors="coerce")
    if "Presion" in df.columns:
        df["Presion"] = pd.to_numeric(df["Presion"], errors="coerce")

    neg_vol = int((df["Volumen"] < 0).sum()) if "Volumen" in df.columns else 0
    neg_pre = int((df["Presion"] < 0).sum()) if "Presion" in df.columns else 0

    if "Volumen" in df.columns:
        df.loc[df["Volumen"] < 0, "Volumen"] = 0.0
    if "Presion" in df.columns:
        df.loc[df["Presion"] < 0, "Presion"] = 0.0

    return df, {"negativos_volumen_a_cero": neg_vol, "negativos_presion_a_cero": neg_pre}


def segmentar_clientes_dbscan(
    df: pd.DataFrame,
    cliente_col: str = "Cliente",
    vol_col: str = "Volumen",
    pres_col: str = "Presion",
) -> pd.DataFrame:
    req = [cliente_col, vol_col, pres_col]
    if any(c not in df.columns for c in req):
        out = df.copy()
        out["Segmento"] = "Comercial"
        return out

    tmp = df[[cliente_col, vol_col, pres_col]].copy()
    tmp[vol_col] = pd.to_numeric(tmp[vol_col], errors="coerce")
    tmp[pres_col] = pd.to_numeric(tmp[pres_col], errors="coerce")
    tmp = tmp.dropna(subset=[cliente_col, vol_col, pres_col])
    if tmp.empty or tmp[cliente_col].nunique() < 2:
        out = df.copy()
        out["Segmento"] = "Comercial"
        return out

    def p90(x):
        return np.nanquantile(x, 0.90)

    perf = (
        tmp.groupby(cliente_col)
           .agg(vol_med=(vol_col, "median"),
                vol_p90=(vol_col, p90),
                pres_med=(pres_col, "median"),
                pres_p90=(pres_col, p90),
                n_obs=(vol_col, "size"))
           .reset_index()
    )

    feats = ["vol_med", "vol_p90", "pres_med", "pres_p90"]
    X = perf[feats].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    db = DBSCAN(eps=0.3425, min_samples=3).fit(Xs)
    labels = db.labels_
    perf["cluster"] = labels

    valid_mask = perf["cluster"] != -1
    if valid_mask.any():
        centers = (
            perf[valid_mask]
            .groupby("cluster")[feats]
            .mean()
            .reset_index()
        )
        centers["score_vol"] = centers["vol_p90"]
        th = centers["score_vol"].median()

        cluster_to_segment = {}
        for _, row in centers.iterrows():
            seg = "Industrial" if row["score_vol"] > th else "Comercial"
            cluster_to_segment[int(row["cluster"])] = seg

        if "Industrial" not in cluster_to_segment.values():
            top = centers.sort_values("score_vol", ascending=False).iloc[0]["cluster"]
            for cid in centers["cluster"]:
                cluster_to_segment[int(cid)] = "Comercial"
            cluster_to_segment[int(top)] = "Industrial"

        perf["Segmento"] = perf["cluster"].map(cluster_to_segment)

        if (~valid_mask).any():
            centers_std = scaler.transform(centers[feats].values)
            for idx in perf.index[~valid_mask]:
                xstd = scaler.transform(perf.loc[[idx], feats].values)
                d = np.linalg.norm(centers_std - xstd, axis=1)
                nearest_c = int(centers.iloc[int(np.argmin(d))]["cluster"])
                perf.loc[idx, "Segmento"] = cluster_to_segment.get(nearest_c, "Comercial")
    else:
        th = perf["vol_p90"].median()
        perf["Segmento"] = np.where(perf["vol_p90"] > th, "Industrial", "Comercial")

    seg_map = dict(zip(perf[cliente_col], perf["Segmento"]))
    out = df.copy()
    out["Segmento"] = out[cliente_col].map(seg_map).fillna("Comercial")
    return out


def process_dataframe(df_all: pd.DataFrame) -> pd.DataFrame:
    ensure_fecha_column(df_all)
    df_all["Fecha"] = parse_dates_series(df_all["Fecha"])

    n_bad = int(df_all["Fecha"].isna().sum())
    if n_bad > 0:
        print(f"[AVISO] {n_bad} registros con 'Fecha' no parseable serán descartados.")
        df_all = df_all.dropna(subset=["Fecha"])

    df_clean, removed = drop_duplicates_by_cliente_fecha(df_all)
    print(f"=== Duplicados por ['Cliente','Fecha'] eliminados: {removed} ===")
    print(f"Total final de registros: {len(df_clean)}")

    df_imputed, rep_imput = imputar_presion_ceros_con_media_prev_3h(df_clean)
    if rep_imput.empty:
        print("=== Imputación de presión ===")
        print("  No se realizaron imputaciones (no se cumplieron condiciones o sin ventana previa).")
    else:
        print("=== Imputación de presión ===")
        for _, r in rep_imput.iterrows():
            avg_val = float(r["presion_imputada_promedio"])
            print(f"  - {r['Cliente']}: {int(r['registros_imputados'])} imputaciones, promedio imputado = {avg_val:.4f}")
        # No guardamos reporte en este modo
        # rep_path = output_csv.parent / "imputacion_presion_report.csv"
        # rep_imput.to_csv(rep_path, index=False, encoding="utf-8-sig")
        # print(f"[OK] Reporte de imputación exportado: {rep_path.resolve()}")

    # === APLICAR CORRECCIÓN DE NEGATIVOS EN VOLUMEN Y PRESIÓN ===
    df_nonneg, negrep = corregir_valores_negativos(df_imputed)
    print("=== Corrección de valores negativos (Volumen/Presion) ===")
    print(f"  - Volumen < 0 llevados a 0: {negrep['negativos_volumen_a_cero']}")
    print(f"  - Presion < 0 llevados a 0: {negrep['negativos_presion_a_cero']}")

    if {"Volumen", "Presion", "Cliente"}.issubset(df_nonneg.columns):
        df_seg = segmentar_clientes_dbscan(df_nonneg)
        seg_counts = df_seg.groupby("Segmento")["Cliente"].nunique()
        print("=== Segmentación de clientes (DBSCAN fijo) ===")
        for seg, cnt in seg_counts.items():
            print(f"  - {seg}: {cnt} clientes")
    else:
        print("[AVISO] No se pudo crear 'Segmento': faltan columnas 'Volumen' y/o 'Presion'.")
        df_seg = df_nonneg.copy()
        df_seg["Segmento"] = "Comercial"

    cols = list(df_seg.columns)
    preferred = ["Fecha", "Cliente", "Segmento"]
    cols = preferred + [c for c in cols if c not in preferred]
    df_seg = df_seg[cols]
    return df_seg

def etl_to_csv(input_xlsx: Path, output_csv: Path) -> int:
    if not input_xlsx.exists():
        print(f"[ERROR] No existe el archivo: {input_xlsx}", file=sys.stderr)
        return 2

    try:
        df_all, per_sheet_counts = load_excel_with_client_column(input_xlsx)
        validate_totals(per_sheet_counts, df_all)
        
        df_processed = process_dataframe(df_all)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] CSV exportado: {output_csv.resolve()} (codificación utf-8-sig)")
        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="etlcontugas.py",
        description="ETL para consolidar Excel multi-hoja de Contugas, imputar presión y segmentar clientes (DBSCAN fijo).",
    )
    p.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        type=str,
        help=f"Ruta al archivo Excel .xlsx de entrada (por defecto: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        type=str,
        help=f"Ruta de salida del CSV (por defecto: {DEFAULT_OUTPUT})",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    in_path = Path(args.input)
    out_path = Path(args.output)

    if out_path.name == DEFAULT_OUTPUT and out_path.parent == Path("."):
        out_path = Path.cwd() / DEFAULT_OUTPUT

    return etl_to_csv(in_path, out_path)


if __name__ == "__main__":
    sys.exit(main())