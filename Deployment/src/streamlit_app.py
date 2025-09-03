# app.py
# Detección de Anomalía - Contugas (CORJ-2514)
# - Carga artefactos por segmento: scaler, ENet (forecast), IsolationForest (residuales)
# - Aplica SSA -> ENet -> residuales -> IF -> fusión (AND/OR)
# - Convención de artefactos: <ruta_base>__seg=<segmento>.pkl
# - Recibe datos de ETL outputs

import os
import io
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ===================== CONFIG DEFAULTS =====================
CFG = {
    "seed": 69069,
    "data": {
        "segment_col": "Segmento",   # <- Debe existir en los datos de entrada
        "client_col": "Cliente",
        "date_col": "Fecha",
        "dayfirst": True
    },
    "paths": {
        # Se usan como BASE y se les agrega __seg=<seg>.pkl
        "enet_model": "./model_outputs/enet_forecast.pkl",
        "scaler": "./model_outputs/forecast_scaler.pkl",
        "iforest": "./model_outputs/iforest_ssa.pkl",
        "eval_dir": "./model_outputs/eval"   # para exportar resultados si se desea en disco
    },
    "ssa": {"window": 24, "rank": 8},
    "regression": {"lags_v": 24, "lags_exo": 12},
    "residuals": {"roll_window": 30},
    "unsupervised": {
        "fusion": "and",        # "and" | "or"
        "z_threshold": 3.5,
        "contamination": 0.03
    }
}

# ===================== UTILITIES (alineadas a tu pipeline) =====================
def _normalize_columns(df: pd.DataFrame, client_hint=None, date_hint=None) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in t if str(x)!='nan']).strip() for t in df.columns.to_list()]
    df.columns = [str(c).replace('\ufeff','').strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    client_aliases = ["cliente","client","clients","idcliente","id_cliente","customer","usuario","meter","medidor"]
    date_aliases   = ["fecha","date","fechahora","fecha_hora","datetime","timestamp","fecha hora"]
    if client_hint: client_aliases = [str(client_hint).lower()] + client_aliases
    if date_hint:   date_aliases   = [str(date_hint).lower()] + date_aliases
    mapping = {}
    for k in client_aliases:
        if k in lower: mapping[lower[k]] = "Cliente"; break
    for k in date_aliases:
        if k in lower: mapping[lower[k]] = "Fecha"; break
    if mapping: df = df.rename(columns=mapping)
    if "Cliente" not in df.columns and "CLIENTE" in df.columns:
        df = df.rename(columns={"CLIENTE":"Cliente"})
    if "Fecha" not in df.columns and "FECHA" in df.columns:
        df = df.rename(columns={"FECHA":"Fecha"})
    return df

def _parse_dates_fixed(df: pd.DataFrame, dayfirst=True) -> pd.DataFrame:
    try:
        df["Fecha"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip(),
            format="%d-%m-%Y %I:%M:%S %p",
            errors="raise"
        )
    except Exception:
        df["Fecha"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip(),
            errors="coerce", dayfirst=dayfirst
        )
    return df

def ssa_decompose(series: np.ndarray, L: int):
    N = len(series)
    if L < 2 or L > N-1:
        raise ValueError(f"L inválido: {L} para longitud {N}")
    K = N - L + 1
    X = np.column_stack([series[i:i+L] for i in range(K)])  # L x K
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.diag(s)
    return U, S, Vt

def ssa_reconstruct_series(U, S, Vt, rank: int):
    U_r = U[:, :rank]; S_r = S[:rank, :rank]; Vt_r = Vt[:rank, :]
    X_r = U_r @ S_r @ Vt_r  # L x K
    L, K = X_r.shape
    N = L + K - 1
    recon = np.zeros(N); counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            recon[i+j] += X_r[i, j]
            counts[i+j] += 1
    recon = recon / np.where(counts == 0, 1, counts)
    return recon

def make_supervised_table(df, lags_v=24, lags_exo=12):
    rows = []
    for cli, g in df.groupby("Cliente"):
        g = g.sort_values("Fecha").reset_index(drop=True)
        maxlag = max(lags_v, lags_exo)
        for t in range(maxlag, len(g)-1):
            row = {"Cliente": cli, "Fecha": g.loc[t, "Fecha"], "y_next": g.loc[t+1, "Volumen"]}
            for k in range(lags_v):
                row[f"Vf_lag{k}"] = g.loc[t-k, "V_filt"]
            for k in range(lags_exo):
                row[f"P_lag{k}"] = g.loc[t-k, "Presion"]
                row[f"T_lag{k}"] = g.loc[t-k, "Temperatura"]
            rows.append(row)
    Xy = pd.DataFrame(rows)
    feat_cols = [c for c in Xy.columns if c not in ("Cliente","Fecha","y_next")]
    return Xy, feat_cols

def residual_features(df_pred, roll_window=30):
    g = df_pred.sort_values(["Cliente","Fecha"]).copy()
    parts = []
    for cli, gi in g.groupby("Cliente"):
        gi = gi.sort_values("Fecha").copy()
        r = gi["resid"].astype(float).values
        rw = max(5, int(roll_window))
        r_mean = pd.Series(r).rolling(rw, min_periods=5).mean().to_numpy()
        r_std  = pd.Series(r).rolling(rw, min_periods=5).std(ddof=0).to_numpy()
        r_std = np.where(np.isnan(r_std) | (r_std == 0), 1e-6, r_std)
        z = (r - np.nan_to_num(r_mean, nan=0.0)) / r_std
        dz = np.diff(np.concatenate([[z[0]], z])).astype(float)
        dr = np.diff(np.concatenate([[r[0]], r])).astype(float)

        out = gi[["Cliente","Fecha"]].copy()
        out["resid"] = r
        out["z_abs"] = np.abs(z)
        out["r_mean"] = np.nan_to_num(r_mean, nan=0.0)
        out["r_std"]  = r_std
        out["dz_abs"] = np.abs(dz)
        out["dr_abs"] = np.abs(dr)
        parts.append(out)
    F = pd.concat(parts, ignore_index=True)
    return F

def _suffix_from_segment(seg):
    s = str(seg)
    for ch in r'\/:*?"<>| ':
        s = s.replace(ch, "_")
    return s

def _derive_segment_paths(base_paths, seg_value):
    seg_suffix = _suffix_from_segment(seg_value)
    out = {}
    for k in ("enet_model", "scaler", "iforest"):
        root, ext = os.path.splitext(base_paths[k])
        out[k] = f"{root}__seg={seg_suffix}{ext}"
    return out

# ===================== CACHED LOADERS =====================
@st.cache_resource(show_spinner=False)
def load_artifacts_for_segment(base_paths: dict, seg_value: str):
    spaths = _derive_segment_paths(base_paths, seg_value)
    missing = [k for k,p in spaths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"No se encuentran artefactos del segmento '{seg_value}'. "
            f"Faltan: {missing}. Se esperan archivos con patrón '__seg={_suffix_from_segment(seg_value)}.pkl'"
        )
    with open(spaths["enet_model"], "rb") as f: enet = pickle.load(f)
    with open(spaths["scaler"], "rb") as f: scaler = pickle.load(f)
    with open(spaths["iforest"], "rb") as f: iso = pickle.load(f)
    return enet, scaler, iso, spaths

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> FIX DE CACHÉ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@st.cache_data(show_spinner=False)
def run_inference_segment(df_seg: pd.DataFrame, C: dict, spaths: dict, _enet=None, _scaler=None, _iso=None):
    """Inferencia cacheada por segmento usando rutas (spaths) como clave hashable.
       Los modelos sklearn se pasan con nombre iniciado en '_' para que Streamlit no los hashee."""
    enet, scaler, iso = _enet, _scaler, _iso

    # ====== SSA por cliente ======
    L    = int(C["ssa"]["window"]); rank = int(C["ssa"]["rank"])
    rows = []
    for cli, g in df_seg.groupby("Cliente"):
        g = g.sort_values("Fecha").copy()
        v = g["Volumen"].astype(float).values
        if len(v) <= L:
            g["V_filt"] = v
        else:
            U,Sv,Vt = ssa_decompose(v, L=L)
            v_rec = ssa_reconstruct_series(U, Sv, Vt, rank=rank)[:len(v)]
            g["V_filt"] = v_rec
        rows.append(g)
    df_ssa = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ====== Features ENet ======
    lags_v   = int(C["regression"]["lags_v"])
    lags_exo = int(C["regression"]["lags_exo"])
    Xy, feat_cols = make_supervised_table(df_ssa, lags_v=lags_v, lags_exo=lags_exo)
    if Xy.empty:
        return pd.DataFrame()  # nada que predecir

    # ====== Forecast + residuales ======
    X_all = scaler.transform(Xy[feat_cols].values.astype(float))
    y_hat = enet.predict(X_all)
    df_pred = Xy[["Cliente","Fecha","y_next"]].copy().rename(columns={"y_next":"Volumen_next"})
    df_pred["Volumen_hat"] = y_hat
    df_pred["resid"] = df_pred["Volumen_next"] - df_pred["Volumen_hat"]

    # ====== Features IF ======
    roll_window = int(C["residuals"]["roll_window"])
    F = residual_features(df_pred, roll_window=roll_window)

    # ====== Scores IF -> [0,1] ======
    X_if_all = F[["z_abs","resid","r_mean","r_std","dz_abs","dr_abs"]].astype(float).values
    dfun  = iso.decision_function(X_if_all)  # alto = normal
    score = -dfun                            # alto = anómalo
    smin, smax = float(np.min(score)), float(np.max(score))
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
    proba = (score - smin) / denom

    # ====== Fusión ======
    fusion = str(C["unsupervised"]["fusion"]).lower()
    z_thr  = float(C["unsupervised"]["z_threshold"])
    cont   = float(C["unsupervised"]["contamination"])
    thr_if = np.quantile(proba, 1.0 - cont)
    flag_if = (proba >= thr_if).astype(int)
    flag_z  = (F["z_abs"].values >= z_thr).astype(int)
    Flag_Final = np.where((flag_if + flag_z) > 0, 1, 0) if fusion == "or" else (flag_if & flag_z).astype(int)

    # ====== Ensamble de salida ======
    out_df = F.copy()
    out_df["proba"] = proba
    out_df["Flag_Final"] = Flag_Final
    out_df = out_df.merge(df_seg[["Cliente","Fecha","Volumen","Presion","Temperatura"]],
                          on=["Cliente","Fecha"], how="left")
    out_df = out_df.merge(df_pred[["Cliente","Fecha","Volumen_next","Volumen_hat"]],
                          on=["Cliente","Fecha"], how="left")
    return out_df.sort_values(["Cliente","Fecha"]).reset_index(drop=True)

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Contugas - Inference (Dev Prod)", layout="wide")
st.title("Contugas · Inference por Segmento (Dev Prod)")

with st.sidebar:
    st.header("Configuración")
    # Opción: cargar YAML externo (opcional). Si no, usa CFG embebido.
    yaml_file = st.file_uploader("Config YAML (opcional)", type=["yaml", "yml"])
    C = CFG.copy()
    if yaml_file is not None:
        import yaml as _yaml
        C = {**CFG, **_yaml.safe_load(yaml_file)}

    # Parámetros clave visibles
    st.markdown("**Artefactos (base):**")
    enet_base   = st.text_input("ENet base",   value=C["paths"]["enet_model"])
    scaler_base = st.text_input("Scaler base", value=C["paths"]["scaler"])
    if_base     = st.text_input("IForest base",value=C["paths"]["iforest"])
    C["paths"]["enet_model"] = enet_base
    C["paths"]["scaler"]     = scaler_base
    C["paths"]["iforest"]    = if_base

    st.markdown("---")
    seg_col = st.text_input("Columna de Segmento", value=C["data"]["segment_col"])
    client_col = st.text_input("Columna de Cliente", value=C["data"]["client_col"])
    date_col = st.text_input("Columna de Fecha", value=C["data"]["date_col"])
    dayfirst = st.checkbox("Fecha con día primero (dd-mm-YYYY...)", value=C["data"]["dayfirst"])

    st.markdown("---")
    st.markdown("**Hiperparámetros (solo lectura)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"SSA L={C['ssa']['window']}, rank={C['ssa']['rank']}")
        st.write(f"lags_v={C['regression']['lags_v']}")
    with col2:
        st.write(f"lags_exo={C['regression']['lags_exo']}")
        st.write(f"roll_window={C['residuals']['roll_window']}")
    with col3:
        st.write(f"fusion={C['unsupervised']['fusion']}")
        st.write(f"z_thr={C['unsupervised']['z_threshold']}, cont={C['unsupervised']['contamination']}")

st.markdown("#### Entrada de datos")
file = st.file_uploader("Sube un CSV o Parquet con columnas mínimas: Segmento, Cliente, Fecha, Volumen, Presion, Temperatura", type=["csv","parquet"])
run_btn = st.button("Ejecutar inferencia")

if run_btn:
    if file is None:
        st.error("Sube un archivo primero.")
        st.stop()

    # ====== Lectura de datos ======
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8-sig")
        else:
            df = pd.read_parquet(file)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    # ====== Normalización de columnas ======
    df = _normalize_columns(df, client_hint=client_col, date_hint=date_col)
    # Renombrar a nombres estandar (por si difieren)
    if client_col in df.columns: df = df.rename(columns={client_col: "Cliente"})
    if date_col   in df.columns: df = df.rename(columns={date_col: "Fecha"})

    # Validaciones mínimas
    req = ["Cliente","Fecha","Volumen","Presion","Temperatura", seg_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {missing}")
        st.stop()

    # Tipos y orden
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=dayfirst)
    for c in ["Volumen","Presion","Temperatura"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Cliente","Fecha","Volumen","Presion","Temperatura"]).sort_values(["Cliente","Fecha"]).reset_index(drop=True)

    segments = [s for s in df[seg_col].dropna().unique().tolist()]
    if len(segments) == 0:
        st.error(f"La columna '{seg_col}' no contiene valores válidos.")
        st.stop()

    st.success(f"Archivo leído. Filas: {len(df):,}. Segmentos detectados: {len(segments)}.")

    # ====== Inferencia por segmento ======
    all_preds = []
    for seg in segments:
        st.markdown(f"### Segmento: `{seg}`")
        df_seg = df[df[seg_col] == seg].copy()
        if df_seg.empty:
            st.info("Sin filas para este segmento.")
            continue

        # Carga de artefactos del segmento
        try:
            enet, scaler, iso, spaths = load_artifacts_for_segment(C["paths"], seg)
        except Exception as e:
            st.error(str(e))
            continue

        # Inferencia (con FIX de caché)
        try:
            preds_df = run_inference_segment(df_seg, C, spaths, _enet=enet, _scaler=scaler, _iso=iso)
        except Exception as e:
            st.error(f"Error de inferencia en segmento {seg}: {e}")
            continue

        if preds_df.empty:
            st.warning("No se generaron filas de predicción (posiblemente series muy cortas para los lags).")
            continue

        # KPIs simples
        anomaly_rate = float(np.mean(preds_df["Flag_Final"])) if len(preds_df) else 0.0
        st.write(f"Filas evaluadas: {len(preds_df):,} · Alarm rate: {anomaly_rate:.4f}")

        # Vista rápida
        st.dataframe(preds_df.head(30))

        # Descarga por segmento
        seg_buf = io.StringIO()
        preds_df.to_csv(seg_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label=f"Descargar CSV (segmento {seg})",
            data=seg_buf.getvalue(),
            file_name=f"preds_inference__seg={_suffix_from_segment(seg)}.csv",
            mime="text/csv"
        )

        all_preds.append(preds_df.assign(**{seg_col: seg}))

    # ====== Consolidado global ======
    if all_preds:
        all_df = pd.concat(all_preds, ignore_index=True)
        st.markdown("### Consolidado (todos los segmentos)")
        st.write(f"Filas totales: {len(all_df):,} · Alarm rate global: {float(np.mean(all_df['Flag_Final'])):.4f}")
        st.dataframe(all_df.head(30))

        all_buf = io.StringIO()
        all_df.to_csv(all_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="Descargar CSV (todos los segmentos)",
            data=all_buf.getvalue(),
            file_name="preds_inference__all_segments.csv",
            mime="text/csv"
        )
    else:
        st.info("No hubo resultados para consolidar.")
