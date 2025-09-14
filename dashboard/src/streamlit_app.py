# -*- coding: utf-8 -*-
# app_streamlit.py — Contugas (SSA + ENet (poly opcional) + Scaler + IForest en residuales), por segmento

import os
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --------------------------------------------------------------------------------
# CONFIG rutas de artefactos: deben existir por segmento con sufijo __seg=<Segmento>.pkl
# --------------------------------------------------------------------------------
BASE_DIR = Path(
    r"E:\Data2\1.Data\2. POSTGRADUATE\11. MAESTRIA_MIAD\ANDES\12_Proyecto_Aplicado_Analítica_de_Datos_(PAAD)\Proyecto_Final_Caso_Contugas\06_Models\Training\model_outputs"
)


# Carpeta del repo "Deployment" (sube 1 nivel desde src/)
ROOT = Path(__file__).resolve().parents[1]

# 1) Carpeta de artefactos (donde están los .pkl)
BASE_DIR = ROOT / "model_outputs"

ARTIFACTS_BY_SEG = {
    "Comercial": {
        "enet":    BASE_DIR / "enet_forecast__seg=Comercial.pkl",
        "scaler":  BASE_DIR / "forecast_scaler__seg=Comercial.pkl",
        "iforest": BASE_DIR / "iforest_ssa__seg=Comercial.pkl",
        "poly":    BASE_DIR / "poly_features__seg=Comercial.pkl",
    },
    "Industrial": {
        "enet":    BASE_DIR / "enet_forecast__seg=Industrial.pkl",
        "scaler":  BASE_DIR / "forecast_scaler__seg=Industrial.pkl",
        "iforest": BASE_DIR / "iforest_ssa__seg=Industrial.pkl",
        "poly":    BASE_DIR / "poly_features__seg=Industrial.pkl",
    },
}

# Hiperparámetros (coherentes con config del training)
SSA_WINDOW = 24
SSA_RANK   = 5
LAGS_V     = 24
LAGS_EXO   = 12
ROLL_Z     = 24
RES_ROLL   = 12
IF_CONTAM  = 0.03
FUSION     = "or"
Z_DEFAULT  = 1e9

# Subset usado para expansión polinomial opcional en el entrenamiento
POLY_V_MAX = 6
POLY_P_MAX = 4
POLY_T_MAX = 4

st.set_page_config(page_title="Gas Contugas - Dashboard", layout="wide")
st.title("📊 Dashboard Para Detección de Anomalías")
st.caption("Históricos, estadísticas, forecasting (SSA+ENet) y anomalías (IForest en residuales) — por segmento")

# --------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------
def load_base_data():
    default_path = "df_contugas.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path, encoding="utf-8-sig")
        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        return df
    return pd.DataFrame(columns=["Fecha","Presion","Temperatura","Volumen","Cliente","Segmento"])

import io

def read_any_file(uploaded_file):
    """Lee CSV/Excel desde st.file_uploader de forma robusta (no consume el buffer)."""
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    # Copiamos los bytes para no consumir el buffer original
    raw = uploaded_file.getvalue()
    if not raw:
        st.error("El archivo subido está vacío.")
        return None
    bio = io.BytesIO(raw)

    if name.endswith((".xlsx", ".xls")):
        bio.seek(0)
        df = pd.read_excel(bio)
    elif name.endswith(".csv"):
        df = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                bio.seek(0)
                # Autodetecta ; o , como separador
                tmp = pd.read_csv(bio, sep=None, engine="python", encoding=enc)
                if tmp.shape[1] == 0:
                    continue
                df = tmp
                break
            except pd.errors.EmptyDataError:
                st.error("El CSV no contiene datos.")
                return None
            except Exception:
                continue
        if df is None:
            st.error("No se pudo leer el CSV. Verifica separador/codificación.")
            return None
    else:
        st.error("Formato no soportado. Sube CSV o Excel.")
        return None

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)

    st.caption(f"📂 Archivo cargado: **{uploaded_file.name}** — {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def ensure_hourly(dfg: pd.DataFrame) -> pd.DataFrame:
    dfg = dfg.sort_values("Fecha").set_index("Fecha").asfreq("H")
    cols = [c for c in ["Presion","Temperatura","Volumen"] if c in dfg.columns]
    if cols:
        dfg[cols] = dfg[cols].interpolate(limit_direction="both")
    dfg = dfg.reset_index()
    return dfg

@st.cache_resource(show_spinner=False)
def load_artifacts(segmento: str):
    paths = ARTIFACTS_BY_SEG[segmento]
    missing = [k for k,p in paths.items() if not Path(p).exists()]
    if "poly" in missing:
        missing.remove("poly")
    if missing:
        raise FileNotFoundError(f"Faltan artefactos para {segmento}: {missing}")
    enet    = joblib.load(paths["enet"])
    scaler  = joblib.load(paths["scaler"])
    iforest = joblib.load(paths["iforest"])
    poly    = joblib.load(paths["poly"]) if Path(paths["poly"]).exists() else None
    return {"enet": enet, "scaler": scaler, "iforest": iforest, "poly": poly}

# ---------------- SSA ----------------
def ssa_decompose(series: np.ndarray, L: int):
    N = len(series)
    if L < 2 or L > max(2, N-1):
        L = max(2, min(L, max(2, N-1)))
    K = N - L + 1
    if K < 1:
        U = np.eye(L)
        s = np.ones(min(L, N))
        Vt = np.eye(min(L, N))
        S = np.diag(s)
        return U, S, Vt
    X = np.column_stack([series[i:i+L] for i in range(K)])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.diag(s)
    return U, S, Vt

def ssa_reconstruct_series(U, S, Vt, rank: int):
    r = max(1, min(rank, min(S.shape)))
    U_r = U[:, :r]; S_r = S[:r, :r]; Vt_r = Vt[:r, :]
    X_r = U_r @ S_r @ Vt_r
    L, K = X_r.shape
    N = L + K - 1
    recon = np.zeros(N); counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            recon[i+j] += X_r[i, j]
            counts[i+j] += 1
    recon = recon / np.where(counts == 0, 1, counts)
    return recon

# ---------------- Tabla supervisada ----------------
def make_supervised_table(df, lags_v=LAGS_V, lags_exo=LAGS_EXO):
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

# ---------------- Residuales -> features IForest ----------------
def residual_features(df_pred: pd.DataFrame, roll_window=RES_ROLL):
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
    F = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Cliente","Fecha","resid","z_abs","r_mean","r_std","dz_abs","dr_abs"])
    return F

# ---------------- Forecast + Anomalías ----------------
def forecast_and_anomalies(dfg: pd.DataFrame, models: dict, z_threshold: float = Z_DEFAULT):
    dfg2 = dfg.sort_values(["Cliente","Fecha"]).copy()
    rows = []
    for cli, g in dfg2.groupby("Cliente"):
        g = g.sort_values("Fecha").copy()
        v = g["Volumen"].astype(float).values
        if len(v) <= SSA_WINDOW:
            g["V_filt"] = v
        else:
            U,S,Vt = ssa_decompose(v, L=SSA_WINDOW)
            v_rec = ssa_reconstruct_series(U, S, Vt, rank=SSA_RANK)[:len(v)]
            g["V_filt"] = v_rec
        rows.append(g)
    df_ssa = pd.concat(rows, ignore_index=True) if rows else dfg2.copy()

    Xy, feat_cols = make_supervised_table(df_ssa, lags_v=LAGS_V, lags_exo=LAGS_EXO)
    if Xy.empty:
        raise ValueError("Muy pocos datos para construir lags y predecir (necesita al menos max(lags_v,lags_exo)+1 filas por cliente).")

    subs_v = [f"Vf_lag{k}" for k in range(POLY_V_MAX)]
    subs_p = [f"P_lag{k}"  for k in range(POLY_P_MAX)]
    subs_t = [f"T_lag{k}"  for k in range(POLY_T_MAX)]
    subset_cols = [c for c in Xy.columns if c in (subs_v + subs_p + subs_t)]

    X_all_base = Xy[feat_cols].values.astype(float)
    if models["poly"] is not None and len(subset_cols) > 0:
        all_subset = Xy[subset_cols].values.astype(float)
        X_all_poly = models["poly"].transform(all_subset)
        X_all_concat = np.concatenate([X_all_base, X_all_poly], axis=1)
    else:
        X_all_concat = X_all_base

    Xs = models["scaler"].transform(X_all_concat)
    y_hat_all = models["enet"].predict(Xs)

    df_pred = Xy[["Cliente","Fecha","y_next"]].copy().rename(columns={"y_next":"Volumen_next"})
    df_pred["Volumen_hat"] = y_hat_all
    df_pred["resid"] = df_pred["Volumen_next"] - df_pred["Volumen_hat"]

    F = residual_features(df_pred, roll_window=RES_ROLL)
    X_if_all = F[["z_abs","resid","r_mean","r_std","dz_abs","dr_abs"]].astype(float).values
    dfun = models["iforest"].decision_function(X_if_all)
    score = -dfun
    smin, smax = float(np.min(score)), float(np.max(score))
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
    proba = (score - smin) / denom

    thr_if  = np.quantile(proba, 1.0 - IF_CONTAM)
    flag_if = (proba >= thr_if).astype(int)
    z_abs   = F["z_abs"].values
    flag_z  = (z_abs >= z_threshold).astype(int)
    flag    = np.where((flag_if + flag_z) > 0, 1, 0) if FUSION == "or" else (flag_if & flag_z).astype(int)

    out = (
        F[["Cliente","Fecha","z_abs","resid","r_mean","r_std","dz_abs","dr_abs"]]
        .assign(Volumen_hat=df_pred["Volumen_hat"].values,
                Volumen_next=df_pred["Volumen_next"].values,
                proba_if=proba,
                Flag_IF=flag_if,
                Flag_Z=flag_z,
                Flag_Final=flag)
        .merge(df_ssa[["Cliente","Fecha","Volumen","Presion","Temperatura","V_filt"]], on=["Cliente","Fecha"], how="left")
        .sort_values(["Cliente","Fecha"])
        .reset_index(drop=True)
    )
    return out, float(thr_if)

# Severidad (sobre proba_if)
def categorize_anomaly_if(p, q_leve=0.70, q_media=0.85, q_critica=0.95):
    if p >= q_critica:
        return "Crítica"
    elif p >= q_media:
        return "Media"
    elif p >= q_leve:
        return "Leve"
    return ""

# --------------------------------------------------------------------------------
# Sidebar — selección y opciones (keys únicas)
# --------------------------------------------------------------------------------
base_df = load_base_data()



#--------------------------------

if "Segmento" not in base_df.columns:
    base_df["Segmento"] = "Comercial"
if "Fecha" in base_df.columns:
    base_df["Fecha"] = pd.to_datetime(base_df["Fecha"], errors="coerce", dayfirst=True)

segmentos_base = (
    sorted(base_df["Segmento"].dropna().astype(str).str.strip().unique().tolist())
    if ("Segmento" in base_df.columns and not base_df.empty)
    else ["Comercial", "Industrial"]
)

st.sidebar.header("Filtros")
segmento_sel_pre = st.sidebar.selectbox(
    "Segmento (para cargar artefactos)",
    options=segmentos_base,
    index=0,
    key="seg_pre"
)


if ("Fecha" in base_df.columns) and base_df["Fecha"].notna().any():
    dmin = pd.to_datetime(base_df["Fecha"].min()).date()
    dmax = pd.to_datetime(base_df["Fecha"].max()).date()
else:
    today = date.today()
    dmin, dmax = today - timedelta(days=30), today

rango = st.sidebar.date_input("Rango de fechas", value=(dmin, dmax), key="rango_fechas")

#--------------------------------

st.sidebar.markdown("---")
uploaded   = st.sidebar.file_uploader("Subir nuevas mediciones (CSV/Excel)", type=["csv","xlsx","xls"], key="file_up")
merge_mode = st.sidebar.radio("¿Cómo usar el archivo subido?", ["Reemplazar rango solapado","Añadir/append"], index=0, key="merge_mode")

# # --------------------------------------------------------------------------------
# # clientes y segmentos
# # --------------------------------------------------------------------------------
df = base_df.copy()
if uploaded is not None:
    new_df = read_any_file(uploaded)
    if new_df is not None and not new_df.empty:
        req_cols = {"Fecha","Volumen","Temperatura","Presion","Cliente"}
        miss = req_cols - set(new_df.columns)
        if miss:
            st.error(f"El archivo subido no contiene columnas requeridas: {sorted(miss)}")
        else:
            new_df = new_df.dropna(subset=["Fecha","Volumen","Temperatura","Presion","Cliente"])
            if "Segmento" not in new_df.columns:
                new_df["Segmento"] = segmento_sel_pre
            if merge_mode == "Reemplazar rango solapado":
                for c, d in new_df.groupby("Cliente"):
                    mn, mx = d["Fecha"].min(), d["Fecha"].max()
                    df = df[~((df["Cliente"]==c) & (df["Fecha"].between(mn, mx)))]
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["Cliente","Fecha"]).reset_index(drop=True)

clientes = sorted(df["Cliente"].astype(str).str.strip().dropna().unique().tolist()) if "Cliente" in df.columns else []
segmentos = sorted(df["Segmento"].astype(str).str.strip().dropna().unique().tolist()) if "Segmento" in df.columns else ["Comercial","Industrial"]

st.sidebar.header("Selección de datos")
cliente_sel = st.sidebar.selectbox("Cliente", options=clientes if clientes else ["—"], index=0, key="cliente_sel")
if not clientes and cliente_sel == "—":
    cliente_sel = None

segmento_sel = st.sidebar.selectbox(
    "Segmento (para cargar artefactos)",
    options=segmentos,
    index=segmentos.index(segmento_sel_pre) if segmento_sel_pre in segmentos else 0,
    key="seg_final"
)

#--------------------------------
st.sidebar.header("Detección de anomalías")
q_leve  = st.sidebar.slider("Severidad Leve IF (≥)",  0.50, 0.999, 0.70, 0.001, format="%.3f", key="q_leve")
q_media = st.sidebar.slider("Severidad Media IF (≥)", 0.50, 0.999, 0.85, 0.001, format="%.3f", key="q_media")
q_crit  = st.sidebar.slider("Severidad Crítica IF (≥)",0.50, 0.999, 0.95, 0.001, format="%.3f", key="q_crit")
if not (q_leve < q_media < q_crit):
    st.sidebar.warning("Orden requerido: Leve < Media < Crítica.")

z_thr = st.sidebar.number_input(
    "Umbral z (|z| ≥) — usa 1e9 para desactivar",
    value=float(Z_DEFAULT),
    step=0.1,
    format="%.1f",
    key="z_thr"
)

# Tolerancia 0–1 (0–100%)
tol_pct = st.sidebar.slider(
    "Tolerancia bandas ± (%)",
    min_value=0.00, max_value=1.00, value=0.80, step=0.01,
    format="%.2f", key="tol_bandas"
)

# Controles de visualización de bandas/anomalías
show_only_out = st.sidebar.checkbox("Mostrar solo puntos fuera de banda", value=False, key="only_oob")
and_iforest   = st.sidebar.checkbox("Cruzar con IForest/Z (Flag_Final)", value=False, key="and_iforest")
marker_size   = st.sidebar.slider("Tamaño puntos rojos", min_value=4, max_value=14, value=6, step=1, key="marker_size")

# --- TAB 1: Histórico + Estadísticas + Anomalías ---
tab1, tab2, tab3 = st.tabs(["📊 Anomalías","📈 Histórico", "👥 Resumen de Clientes"])

# -----------------------------
# TAB 1: Anomalías
# -----------------------------

with tab1:
    # Copia base y (opcional) merge con archivo subido
    df = base_df.copy()
    if uploaded is not None:
        new_df = read_any_file(uploaded)
        if new_df is not None and not new_df.empty:
            req_cols = {"Fecha","Volumen","Temperatura","Presion","Cliente"}
            miss = req_cols - set(new_df.columns)
            if miss:
                st.error(f"El archivo subido no contiene columnas requeridas: {sorted(miss)}")
            else:
                new_df = new_df.dropna(subset=["Fecha","Volumen","Temperatura","Presion","Cliente"])
                if "Segmento" not in new_df.columns:
                    new_df["Segmento"] = segmento_sel_pre
                if merge_mode == "Reemplazar rango solapado":
                    for c, d in new_df.groupby("Cliente"):
                        mn, mx = d["Fecha"].min(), d["Fecha"].max()
                        df = df[~((df["Cliente"]==c) & (df["Fecha"].between(mn, mx)))]
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    df = pd.concat([df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=["Cliente","Fecha"]).reset_index(drop=True)

    if not df.empty and cliente_sel:
        dfg = df[df["Cliente"]==cliente_sel].copy()

        # Fechas y muestreo horario
        if "Fecha" in dfg.columns:
            dfg["Fecha"] = pd.to_datetime(dfg["Fecha"], errors="coerce")
            dfg = dfg.dropna(subset=["Fecha"])
        dfg = ensure_hourly(dfg)

        # Filtro por rango del sidebar
        if isinstance(rango, (list, tuple)) and len(rango)==2 and rango[0] and rango[1]:
            dfg = dfg[(dfg["Fecha"]>=pd.to_datetime(rango[0])) & (dfg["Fecha"]<=pd.to_datetime(rango[1]))]

            # ==========================================
            # 3) Detección de anomalías (Pred vs Real)
            # ==========================================
            st.subheader("🚨 Detección de anomalías")
            try:
                # Cargar artefactos del segmento
                seg_for_models = segmento_sel
                if "Segmento" in dfg.columns:
                    top = dfg["Segmento"].mode()
                    if not top.empty:
                        seg_for_models = top.iloc[0]
                models = load_artifacts(seg_for_models)

                # Auditoría opcional
                with st.expander("🔎 Auditoría de artefactos (debug)"):
                    st.write({
                        "poly.exists": models["poly"] is not None,
                        "poly.n_features_in_": getattr(models["poly"], "n_features_in_", None) if models["poly"] is not None else None,
                        "poly.n_output_features_": getattr(models["poly"], "n_output_features_", None) if models["poly"] is not None else None,
                        "scaler.n_features_in_": getattr(models["scaler"], "n_features_in_", None),
                        "enet.n_features_in_": getattr(models["enet"], "n_features_in_", None),
                        "iforest.n_features_in_": getattr(models["iforest"], "n_features_in_", None),
                    })

                # Predicción y anomalías
                anomalies, thr_if = forecast_and_anomalies(dfg, models, z_threshold=float(z_thr))

                # ---- Gráfico: Pred vs Real
                df_plot = anomalies.rename(columns={"Volumen_next":"Volumen_real", "Volumen_hat":"Volumen_pred"})
                fig_pred = px.line(
                    df_plot.melt(id_vars="Fecha", value_vars=["Volumen_real","Volumen_pred"],
                                 var_name="serie", value_name="valor"),
                    x="Fecha", y="valor", color="serie",
                    color_discrete_map={"Volumen_real": "#1f77b4", "Volumen_pred": "#636EFA"},
                    title="Predicción vs Real (horizonte t+1)"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # ---- Gráfico: Anomalías detectadas (puntos) usando bandas ±tol y Flag_Final
                subs = df_plot.dropna(subset=["Volumen_real","Volumen_pred"]).copy()
                tol = float(tol_pct)  # 0–1
                subs["upper"] = subs["Volumen_pred"] * (1.0 + tol)
                subs["lower"] = subs["Volumen_pred"] * (1.0 - tol)
                subs["out_of_band"] = ((subs["Volumen_real"] > subs["upper"]) |
                                       (subs["Volumen_real"] < subs["lower"])).astype(int)
                if and_iforest:
                    subs["out_of_band"] = (subs["out_of_band"].astype(bool) &
                                           (subs["Flag_Final"] == 1)).astype(int)

                fig_pts = go.Figure()
                # líneas finas (opcionalmente atenuadas si solo quieres ver puntos)
                opacity_lines = 0.5 if show_only_out else 1.0
                fig_pts.add_trace(go.Scatter(x=subs["Fecha"], y=subs["Volumen_real"],
                                             name="Volumen_real", mode="lines",
                                             line=dict(width=1), opacity=opacity_lines))
                fig_pts.add_trace(go.Scatter(x=subs["Fecha"], y=subs["Volumen_pred"],
                                             name="Volumen_pred", mode="lines",
                                             line=dict(width=1), opacity=opacity_lines))
                # puntos rojos
                an = subs[subs["out_of_band"] == 1]
                if not an.empty:
                    fig_pts.add_trace(go.Scatter(
                        x=an["Fecha"], y=an["Volumen_real"], mode="markers",
                        name="Anomalías (puntos)",
                        marker=dict(color="red", size=int(marker_size)),
                        hovertemplate=("Fecha=%{x}<br>Real=%{y:.3f}"
                                       "<br>Pred=%{customdata[0]:.3f}"
                                       "<br>Upper=%{customdata[1]:.3f}"
                                       "<br>Lower=%{customdata[2]:.3f}<extra></extra>"),
                        customdata=np.stack([an["Volumen_pred"], an["upper"], an["lower"]], axis=1)
                    ))
                fig_pts.update_layout(title=f"Anomalías detectadas (puntos) — bandas ±{tol*100:.1f}%")
                st.plotly_chart(fig_pts, use_container_width=True)

                # Descarga CSV de puntos fuera de banda
                if not an.empty:
                    csv_bytes = an[["Cliente","Fecha","Volumen_real","Volumen_pred","upper","lower","Flag_Final","proba_if"]]\
                                .to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button("⬇️ Descargar anomalías (puntos fuera de banda)",
                                       data=csv_bytes, file_name="anomalies_out_of_band.csv",
                                       mime="text/csv")

                # Métricas rápidas
                subs_err = df_plot.dropna(subset=["Volumen_real","Volumen_pred"]).copy()
                if not subs_err.empty:
                    err  = subs_err["Volumen_real"].values - subs_err["Volumen_pred"].values
                    mae  = float(np.mean(np.abs(err)))
                    rmse = float(np.sqrt(np.mean(err**2)))
                    denom = np.maximum(np.abs(subs_err["Volumen_real"].values), 1e-8)
                    mape = float(np.mean(np.abs(err/denom))*100.0)
                    st.caption(f"MAE={mae:,.3f} | RMSE={rmse:,.3f} | MAPE={mape:,.2f}%  (t+1)")

                st.dataframe(df_plot.tail(300), use_container_width=True)
                st.caption(
                    f"IForest: percentil **{1.0-IF_CONTAM:.3f}** ⇒ score ≥ **{thr_if:.3f}** (Flag_IF=1). "
                    f"Severidad (sobre proba_if): Leve ≥ {q_leve:.3f}, Media ≥ {q_media:.3f}, Crítica ≥ {q_crit:.3f}."
                )

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)

# -----------------------------
# TAB 2: Historioco
# -----------------------------
with tab2:
    st.header("📈 Histórico")
# =========================
# 1) Gráficos históricos (apilados)
# =========================

    if not df.empty and cliente_sel:

        dfg = df[df["Cliente"]==cliente_sel].copy()
        dfg = ensure_hourly(dfg)
        
        if isinstance(rango, (list, tuple)) and len(rango)==2 and rango[0] and rango[1]:
            dfg = dfg[(dfg["Fecha"]>=pd.to_datetime(rango[0])) & (dfg["Fecha"]<=pd.to_datetime(rango[1]))]


        # Preparar agregados diarios (útiles para promedios diarios y percentiles)
        tmp = dfg.set_index("Fecha").sort_index()
        daily = tmp[["Volumen", "Presion", "Temperatura"]].resample("D").agg({
            "Volumen": "sum",        # consumo diario total (suma de horas)
            "Presion": "mean",       # promedio diario de presión
            "Temperatura": "mean"    # promedio diario de temperatura
        })

        # Calcular métricas
        # num_clients = int(dfg["Cliente"].nunique())
        total_consumo = float(dfg["Volumen"].sum()) if "Volumen" in dfg.columns else float("nan")
        avg_daily_vol = float(daily["Volumen"].mean()) if not daily["Volumen"].empty else float("nan")
        avg_daily_presion = float(daily["Presion"].mean()) if not daily["Presion"].empty else float("nan")
        avg_daily_temp = float(daily["Temperatura"].mean()) if not daily["Temperatura"].empty else float("nan")
        p90_vol = float(daily["Volumen"].quantile(0.9)) if not daily["Volumen"].empty else float("nan")

        # Formateo amigable
        def fmt_num(x, dp=2):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "N/A"
            # Si es entero grande, mostrar sin decimales
            if abs(x) >= 1000 and float(x).is_integer():
                return f"{int(x):,}"
            return f"{x:,.{dp}f}"

        # Mostrar 2 filas x 3 columnas con st.metric (cards)
        r1c2, r1c3 = st.columns(2)
        r2c1, r2c2, r2c3 = st.columns(3)

        
        with r1c2:
            st.metric("Consumo total (m³)", fmt_num(total_consumo))
        with r1c3:
            st.metric("Promedio diario — Volumen (m³)", fmt_num(avg_daily_vol))
        with r2c1:
            st.metric("Promedio diario — Presión", fmt_num(avg_daily_presion, dp=3))
        with r2c2:
            st.metric("Promedio diario — Temperatura (°C)", fmt_num(avg_daily_temp, dp=2))
        with r2c3:
            st.metric("P90 diario — Volumen (m³)", fmt_num(p90_vol))


#--------------------------------
    st.subheader("📉 Volumen (histórico)")
    fig_vol = px.line(dfg, x="Fecha", y="Volumen", color_discrete_sequence=["#1f77b4"])
    fig_vol.update_layout(xaxis_title="Fecha", yaxis_title="Volumen (m³)", height=340)
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()  # separador opcional

    st.subheader("🌡️ Temperatura (histórico)")
    fig_tmp = px.line(dfg, x="Fecha", y="Temperatura", color_discrete_sequence=["#e45756"])
    fig_tmp.update_layout(xaxis_title="Fecha", yaxis_title="Temperatura (°C)", height=340)
    st.plotly_chart(fig_tmp, use_container_width=True)

    st.divider()  # separador opcional

    st.subheader("⚙️ Presión (histórico)")
    fig_prs = px.line(dfg, x="Fecha", y="Presion", color_discrete_sequence=["#2ca02c"])
    fig_prs.update_layout(xaxis_title="Fecha", yaxis_title="Presión (bar)", height=340)
    st.plotly_chart(fig_prs, use_container_width=True)

# ==========================================
# 2) Estadísticas descriptivas (rango filtro)
# ==========================================
    st.subheader("📈 Estadísticas descriptivas (rango filtrado)")
    stats_cols = [c for c in ["Volumen","Temperatura","Presion"] if c in dfg.columns]
    if stats_cols:
        st.dataframe(dfg[stats_cols].describe(percentiles=[0.25,0.5,0.75]).T,
            use_container_width=True)


# -----------------------------
# TAB 3: Resumen de Clientes
# -----------------------------

with tab3:
    st.header("👥 Resumen General de Clientes")

    # Top clientes por consumo
    top_clientes = df.groupby("Cliente")["Volumen"].sum().nlargest(10).reset_index()
    st.subheader("🏆 Top 10 clientes por consumo")
    fig_top = px.bar(
        top_clientes,
        x="Cliente",
        y="Volumen",
        text="Volumen",
        color="Volumen",
        color_continuous_scale="Blues",
        title="Clientes con mayor consumo"
    )
    fig_top.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig_top.update_layout(xaxis_title="Cliente", yaxis_title="Volumen (m³)")
    st.plotly_chart(fig_top, use_container_width=True)

    # Distribución por segmento
    st.subheader("📊 Distribución y estadísticas por segmento")
    c1, c2 = st.columns(2)

    # Distribución (pie chart)
    with c1:
        fig_seg = px.pie(
            df,
            names="Segmento",
            title="Distribución de clientes por segmento",
            hole=0.4
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    # Estadísticas por segmento (barras)
    with c2:
        seg_stats = df.groupby("Segmento")["Volumen"].agg(
            Consumo_Total="sum",
            Consumo_Promedio="mean"
        ).reset_index()

        fig_seg_stats = px.bar(
            seg_stats,
            x="Segmento",
            y="Consumo_Total",
            text="Consumo_Total",
            color="Consumo_Total",
            color_continuous_scale="Viridis",
            title="Consumo total por segmento"
        )
        fig_seg_stats.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_seg_stats.update_layout(xaxis_title="Segmento", yaxis_title="Volumen (m³)")
        st.plotly_chart(fig_seg_stats, use_container_width=True)

    # Estadísticas generales
    st.subheader("📌 Estadísticas generales")
    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes únicos", df["Cliente"].nunique())
    col2.metric("Consumo total", f"{df['Volumen'].sum():,.0f} m³")
    col3.metric("Consumo promedio", f"{df['Volumen'].mean():,.2f} m³")
    

# Footer
st.caption("Artefactos y lógica replican el pipeline SSA→lags→(poly subset)→Scaler→ENet→residuales→IForest. "
           "Usa datos con columnas: Fecha, Volumen, Presion, Temperatura, Cliente, Segmento.")
