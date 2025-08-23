
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

st.title("📊 Dashboard de Consumo de Gas (2019-2020)")
st.caption("Históricos, estadísticas, detección de anomalías y validación con LSTM")

# -----------------------------
# Helpers
# -----------------------------
def load_base_data():
    # Default bundled dataset path (can be replaced by your own)
    default_path = "synthetic_training.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    return pd.DataFrame(columns=["Fecha","Presion","Temperatura","Volumen","Cliente"])

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

def notify_slack(webhook_url, message):
    try:
        r = requests.post(webhook_url, json={"text": message}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def notify_email(host, port, username, password, to_email, subject, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = username
        msg["To"] = to_email
        s = smtplib.SMTP(host, port, timeout=5)
        s.starttls()
        s.login(username, password)
        s.sendmail(username, [to_email], msg.as_string())
        s.quit()
        return True
    except Exception:
        return False

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

st.sidebar.header("Notificaciones (opcional)")
slack_url = st.sidebar.text_input("Slack Webhook URL", value="")
email_host = st.sidebar.text_input("SMTP Host", value="")
email_port = st.sidebar.number_input("SMTP Port", value=587)
email_user = st.sidebar.text_input("SMTP User", value="")
email_pass = st.sidebar.text_input("SMTP Password", value="", type="password")
email_to = st.sidebar.text_input("Enviar alertas a (email)", value="")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Subir nuevas mediciones (CSV/Excel)", type=["csv","xlsx","xls"])
merge_mode = st.sidebar.radio("¿Cómo usar el archivo subido?", options=["Reemplazar rango solapado","Añadir/append"], index=0)

# -----------------------------
# Datos (merge si sube archivo)
# -----------------------------
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

    # ---------------------------------
    # Req 1: Gráficos históricos
    # ---------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Volumen (histórico)")
        st.plotly_chart(px.line(dfg, x="Fecha", y="Volumen", title=None), use_container_width=True)
    with c2:
        st.subheader("Temperatura (histórico)")
        st.plotly_chart(px.line(dfg, x="Fecha", y="Temperatura", title=None), use_container_width=True)
    with c3:
        st.subheader("Presión (histórico)")
        st.plotly_chart(px.line(dfg, x="Fecha", y="Presion", title=None), use_container_width=True)

    # ---------------------------------
    # Req 2: Estadísticas descriptivas
    # ---------------------------------
    st.subheader("📈 Estadísticas descriptivas (rango filtrado)")
    stats = dfg[["Volumen","Temperatura","Presion"]].describe(percentiles=[0.25,0.5,0.75]).T
    st.dataframe(stats, use_container_width=True)

    # ---------------------------------
    # Req 3 & 5: Detección de anomalías + nivel (con apoyo del LSTM)
    # ---------------------------------
    st.subheader("🚨 Detección de anomalías (validación con LSTM)")
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

        # ---------------------------------
        # Req 6: Notificaciones
        # ---------------------------------
        if not anom_points.empty:
            msg = f"[{cliente_sel}] {len(anom_points)} anomalías (>= leve) entre {anom_points['Fecha'].min()} y {anom_points['Fecha'].max()}."
            colA, colB = st.columns(2)
            with colA:
                if slack_url:
                    ok = notify_slack(slack_url, msg)
                    st.success("Notificado a Slack ✅" if ok else "No se pudo notificar a Slack ❌")
            with colB:
                if email_host and email_user and email_pass and email_to:
                    ok = notify_email(email_host, int(email_port), email_user, email_pass, email_to, "Alerta de anomalías", msg)
                    st.success("Correo enviado ✅" if ok else "No se pudo enviar correo ❌")
        else:
            st.info("No se detectaron anomalías según los umbrales definidos.")

# Footer
st.caption("Tip: Entrena modelos con `python train_lstm.py --data data/synthetic_training.csv` y reinicia la app para cargarlos.")
