
import os
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Optional notifications
import requests
import smtplib
from email.mime.text import MIMEText

# TensorFlow/Keras
from tensorflow import keras

st.set_page_config(page_title="Gas Industrial - Dashboard", layout="wide")

st.title("📊 Dashboard Contugas")
st.caption("Análisis histórico y detección de anomalías")

# -----------------------------
# Helpers
# -----------------------------
def load_base_data():
    default_path = "data/init_data.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)

        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"])

        expected_cols = ["Fecha", "Cliente", "Segmento", "Presion", "Temperatura", "Volumen"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas en el dataset: {missing}")
            return pd.DataFrame(columns=expected_cols)
        
        return df

    # If file does not exist, return empty dataframe with expected columns
    return pd.DataFrame(columns=["Fecha","Cliente","Segmento","Presion","Temperatura","Volumen"])

def read_any_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Formato no soportado. Sube CSV o Excel.")
        return None
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df

def ensure_hourly(dfg):
    dfg = dfg.sort_values("Fecha").set_index("Fecha")
    dfg = dfg.asfreq("H")
    dfg[["Presion","Temperatura","Volumen"]] = dfg[["Presion","Temperatura","Volumen"]].interpolate(limit_direction="both")
    dfg = dfg.reset_index()
    return dfg

def load_model_for_client(client):
    meta_path = "models/metadata.json"
    if not os.path.exists(meta_path):
        return None, None, None
    meta = json.load(open(meta_path))
    if client not in meta:
        return None, None, None
    model = keras.models.load_model(meta[client]["model_path"])
    lookback = meta[client]["lookback"]
    # Load scalers
    sc = np.load(f"models/{client}_scaler.npz")
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    # Rebuild scaler states
    X_scaler.min_ = sc["X_min"]; X_scaler.scale_ = sc["X_scale"]; X_scaler.data_min_ = sc["X_data_min"]; X_scaler.data_max_ = sc["X_data_max"]; X_scaler.n_features_in_ = len(X_scaler.data_min_)
    y_scaler.min_ = sc["y_min"]; y_scaler.scale_ = sc["y_scale"]; y_scaler.data_min_ = sc["y_data_min"]; y_scaler.data_max_ = sc["y_data_max"]; y_scaler.n_features_in_ = 1
    return model, X_scaler, y_scaler, lookback

def make_sequences_np(X, lookback):
    Xs = []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
    return np.array(Xs)

def categorize_anomaly(z, z_low=2.0, z_mid=3.0, z_high=4.0):
    if abs(z) >= z_high:
        return "Crítica"
    elif abs(z) >= z_mid:
        return "Media"
    elif abs(z) >= z_low:
        return "Leve"
    return ""

# -----------------------------
# Sidebar - filtros y config
# -----------------------------
base_df = load_base_data()

clientes = sorted(base_df["Cliente"].unique().tolist()) if not base_df.empty else []
st.sidebar.header("Filtros")
cliente_sel = st.sidebar.selectbox("Cliente", options=clientes)
rango = st.sidebar.date_input("Rango de fechas", value=(base_df["Fecha"].min().date() if not base_df.empty else None,
                                                        base_df["Fecha"].max().date() if not base_df.empty else None))

st.sidebar.header("Detección de anomalías")
z_low = st.sidebar.number_input("Umbral Leve (|z| ≥)", value=2.0, step=0.1)
z_mid = st.sidebar.number_input("Umbral Media (|z| ≥)", value=3.0, step=0.1)
z_high = st.sidebar.number_input("Umbral Crítica (|z| ≥)", value=4.0, step=0.1)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Subir nuevas mediciones (CSV/Excel)", type=["csv","xlsx","xls"])
merge_mode = st.sidebar.radio("¿Cómo usar el archivo subido?", options=["Reemplazar rango solapado","Añadir/append"], index=0)

# -----------------------------
# Datos (merge si sube archivo)
# -----------------------------
tab1, tab2 = st.tabs(["📈 Histórico", "👥 Resumen de Clientes"])

with tab1:

    df = base_df.copy()
    if uploaded is not None:
        new_df = read_any_file(uploaded)
        if new_df is not None and not new_df.empty:
            new_df = new_df.dropna(subset=["Fecha","Volumen","Temperatura","Presion","Cliente"])
            # Opcional: mismo cliente o múltiples
            if merge_mode == "Reemplazar rango solapado":
                # Eliminar del base el rango que se solape con el subido por cliente
                for c, d in new_df.groupby("Cliente"):
                    mn, mx = d["Fecha"].min(), d["Fecha"].max()
                    df = df[~((df["Cliente"]==c) & (df["Fecha"].between(mn, mx)))]
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["Cliente","Fecha"]).reset_index(drop=True)

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
        num_clients = int(dfg["Cliente"].nunique())
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
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)

        with r1c1:
            st.metric("Número de clientes", f"{num_clients}")
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

        # ---------------------------------
        # Req 1: Gráficos históricos
        # ---------------------------------

        # Volumen
        st.subheader("📉 Volumen (histórico)")
        fig_vol = px.line(
            dfg,
            x="Fecha",
            y="Volumen",
            title="Evolución del Volumen de Consumo",
            line_shape="linear"
        )
        fig_vol.update_traces(line=dict(width=2, color="#1f77b4"))
        fig_vol.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Volumen (m³)",
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, zeroline=False),
            height=400
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Temperatura
        st.subheader("🌡️ Temperatura (histórico)")
        fig_temp = px.line(
            dfg,
            x="Fecha",
            y="Temperatura",
            title="Evolución de la Temperatura",
            line_shape="linear"
        )
        fig_temp.update_traces(line=dict(width=2, color="#e45756"))
        fig_temp.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Temperatura (°C)",
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, zeroline=False),
            height=400
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # Presión
        st.subheader("⚙️ Presión (histórico)")
        fig_pres = px.line(
            dfg,
            x="Fecha",
            y="Presion",
            title="Evolución de la Presión",
            line_shape="linear"
        )
        fig_pres.update_traces(line=dict(width=2, color="#2ca02c"))
        fig_pres.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Presión (bar)",
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, zeroline=False),
            height=400
        )
        st.plotly_chart(fig_pres, use_container_width=True)

        # ---------------------------------
        # Req 2: Estadísticas descriptivas
        # ---------------------------------
        st.subheader("📈 Estadísticas descriptivas (rango filtrado)")
        stats = dfg[["Volumen","Temperatura","Presion"]].describe(percentiles=[0.25,0.5,0.75]).T
        st.dataframe(stats, use_container_width=True)

        # ---------------------------------
        # Req 3 & 5: Detección de anomalías + nivel (con apoyo del LSTM)
        # ---------------------------------
        st.subheader("🚨 Detección de anomalías")
        model, X_scaler, y_scaler, lookback = load_model_for_client(cliente_sel)
        if model is None:
            st.info("No se encontró un modelo entrenado para este cliente. Entrena primero con `train_lstm.py`.")
        else:
            feats = ["Volumen","Temperatura","Presion"]
            dfg2 = dfg.sort_values("Fecha").copy()
            X = dfg2[feats].values
            # Escala con scaler entrenado
            X_scaled = X_scaler.transform(X)
            Xs = []
            ts_list = []
            for i in range(len(X_scaled) - lookback):
                Xs.append(X_scaled[i:i+lookback])
                ts_list.append(dfg2.iloc[i+lookback]["Fecha"])
            Xs = np.array(Xs)

            y_pred_scaled = model.predict(Xs, verbose=0)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

            # Alinear con valores reales (Volumen)
            y_true = dfg2["Volumen"].iloc[lookback:].values
            ts = pd.to_datetime(pd.Series(ts_list))

            resid = y_true - y_pred
            # Z-score sobre residuo (rolling robusto opcional)
            resid_mean = pd.Series(resid).rolling(24, min_periods=12).mean()
            resid_std = pd.Series(resid).rolling(24, min_periods=12).std().replace(0, np.nan)
            z = (pd.Series(resid) - resid_mean) / resid_std
            z = z.fillna(0.0)

            anomalies = pd.DataFrame({
                "Fecha": ts,
                "Cliente": cliente_sel,
                "Volumen_real": y_true,
                "Volumen_pred": y_pred,
                "Residuo": resid,
                "zscore": z
            })
            anomalies["Nivel"] = anomalies["zscore"].apply(lambda v: categorize_anomaly(v, z_low, z_mid, z_high))

            st.plotly_chart(px.line(anomalies, x="Fecha", y=["Volumen_real","Volumen_pred"], title="Predicción vs Real"), use_container_width=True)
            # Puntos de anomalía
            anom_points = anomalies[anomalies["Nivel"] != ""]
            if not anom_points.empty:
                sc = px.scatter(anom_points, x="Fecha", y="Volumen_real", hover_data=["Nivel","zscore"], title="Anomalías detectadas (puntos)")
                st.plotly_chart(sc, use_container_width=True)

            st.dataframe(anomalies.tail(200), use_container_width=True)

with tab2:
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
st.caption("Tip: Entrena modelos con `python train_lstm.py --data data/synthetic_training.csv` y reinicia la app para cargarlos.")
