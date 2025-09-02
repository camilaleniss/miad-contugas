# *** Contugas: SSA (Volumen) + ENet (forecast) + IsolationForest en residuales ***
#            ENTRENANDO POR SEGMENTO (p.ej., Comercial / Industrial)
#
# - Sin cambios de hiperparámetros: se reusa exactamente la misma config.
# - Nuevo: loop por segmento (data.segment_col, default "Segmento").
# - Se guardan modelos/artefactos por segmento con sufijos: __seg=<segment>.
#
# - Corrección MLflow: Todas las métricas/params se loguean en el run maestro.
#   (Se eliminan start_run(..., nested=True) y end_run() dentro del loop por segmento.)

import os, sys, subprocess, importlib, pickle, warnings
import numpy as np
import pandas as pd

def ensure_requirements(reqs=(
    "numpy","pandas","scikit-learn","pyarrow","mlflow","pyyaml"
)):
    for pkg in reqs:
        mod = pkg.split("==")[0].split(">=")[0].split("[")[0].replace("-", "_")
        try:
            importlib.import_module(mod)
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception as _e:
                print(f"[WARN] Could not auto-install {pkg}: {_e}")

ensure_requirements()

try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning

try:
    import mlflow
except Exception:
    mlflow = None

# ---------------- utilidades ----------------
def _safe_make_dirs():
    os.makedirs("model_outputs", exist_ok=True)

def _normalize_columns(df: pd.DataFrame, client_hint=None, date_hint=None) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in t if str(x)!='nan']).strip() for t in df.columns.to_list()]
    df.columns = [str(c).replace('\\ufeff','').strip() for c in df.columns]
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

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["Fecha"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip(),
            format="%d-%m-%Y %I:%M:%S %p",
            errors="raise"
        )
    except Exception:
        df["Fecha"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip(),
            errors="coerce", dayfirst=True
        )
    return df

# ---------------- SSA univariado ----------------
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

# ---------------- Features supervisadas para forecast ----------------
def make_supervised_table(df, lags_v=24, lags_exo=3):
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

def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# ---------------- Residuales -> features no supervisado ----------------
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
    # Limpia el nombre para usarlo en paths
    s = str(seg)
    for ch in r'\/:*?"<>| ':
        s = s.replace(ch, "_")
    return s

def main(cfg_path="config_ssa.yaml"):
    global mlflow

    # --- Config ---
    with open(cfg_path, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    seed = int(C.get("seed", 69069))
    np.random.seed(seed)

    T = C.get("training", {})
    if bool(T.get("suppress_warnings", True)):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # --- MLflow (inicio de run maestro) ---
    mlconf = C.get("mlflow", {})
    ml_enabled = bool(mlconf.get("enabled", True) and (mlflow is not None))
    if ml_enabled:
        mlflow.set_tracking_uri(mlconf.get("tracking_uri", os.environ.get("MLFLOW_TRACKING_URI","file:./mlruns")))
        mlflow.set_experiment(mlconf.get("experiment_name", "Contugas-Anomalias"))
        mlflow.start_run(run_name=mlconf.get("run_name", "run_pipeline_ssa_v01"))
        mlflow.set_tag("training_mode", "per-segment_master_logs")

    # --- DATA ---
    D = C.get("data", {})
    csv_path = D.get("csv_path","./df_contugas.csv")
    seg_col  = D.get("segment_col","Segmento")
    df = pd.read_csv(csv_path, encoding=D.get("encoding","utf-8-sig"))

    # Log del dataset de entrada
    if ml_enabled:
        try:
            ds = mlflow.data.from_pandas(df, source=csv_path, name="contugas_raw")
            mlflow.log_input(ds, context="training")
        except Exception:
            mlflow.set_tag("dataset_name", os.path.basename(csv_path))
            try:
                mlflow.log_artifact(csv_path)
            except Exception:
                pass

    # Normaliza nombres clave y parsea fechas
    df = _normalize_columns(df, client_hint=D.get("client_col"), date_hint=D.get("date_col"))
    if D.get("client_col","Cliente") in df.columns:
        df = df.rename(columns={D.get("client_col","Cliente"):"Cliente"})
    if D.get("date_col","Fecha") in df.columns:
        df = df.rename(columns={D.get("date_col","Fecha"):"Fecha"})
    if "Cliente" not in df.columns or "Fecha" not in df.columns:
        raise KeyError(f"No encuentro columnas Cliente/Fecha. Columnas: {list(df.columns)}")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=bool(D.get("dayfirst", True)))

    req = ["Cliente","Fecha","Volumen","Presion","Temperatura"]
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Faltan columnas requeridas: {miss}")
    for c in ["Volumen","Presion","Temperatura"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=req).sort_values(["Cliente","Fecha"]).reset_index(drop=True)

    # --- Verifica columna de segmento ---
    if seg_col not in df.columns:
        raise KeyError(f"No encuentro la columna de segmento '{seg_col}'. "
                       f"Añádela al CSV o define data.segment_col en el YAML.")

    segments = [s for s in df[seg_col].dropna().unique().tolist()]
    if len(segments) == 0:
        raise ValueError(f"La columna '{seg_col}' no tiene valores válidos.")

    _safe_make_dirs()

    # === LOOP POR SEGMENTO ===
    for seg in segments:
        seg_suffix = _suffix_from_segment(seg)
        df_seg = df[df[seg_col] == seg].copy()
        if df_seg.empty:
            continue

        if ml_enabled:
            # Sólo etiquetamos el segmento en el run maestro
            mlflow.set_tag("segment_current", str(seg))

        # --- SSA por cliente (dentro del segmento) ---
        S = C.get("ssa", {})
        L = int(S.get("window", 24))
        rank = int(S.get("rank", 8))
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

        # --- Tabla supervisada para forecast (ENet) ---
        R = C.get("regression", {})
        lags_v   = int(R.get("lags_v", 24))
        lags_exo = int(R.get("lags_exo", 12))
        Xy, feat_cols = make_supervised_table(df_ssa, lags_v=lags_v, lags_exo=lags_exo)

        # split temporal
        test_size = float(R.get("test_size", 0.40))
        n = len(Xy)
        if n < 10:
            if ml_enabled: mlflow.set_tag("segment_skip_reason", f"too_few_samples__seg={seg_suffix}")
            continue
        cut = max(1, int(n * (1.0 - test_size)))
        train = Xy.iloc[:cut].copy()
        val   = Xy.iloc[cut:].copy()

        # Escalado
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train = scaler.fit_transform(train[feat_cols].values.astype(float))
        y_train = train["y_next"].values.astype(float)
        X_val   = scaler.transform(val[feat_cols].values.astype(float))
        y_val   = val["y_next"].values.astype(float)

        cap = int(T.get("max_train_samples", 0)) or None
        if cap is not None and len(X_train) > cap:
            X_train = X_train[-cap:]; y_train = y_train[-cap:]

        # ENet
        enet = ElasticNet(alpha=float(R.get("alpha", 0.2)), l1_ratio=float(R.get("l1_ratio", 0.1)),
                          max_iter=int(R.get("max_iter", 10000)), tol=float(R.get("tol", 1e-3)),
                          random_state=seed)
        enet.fit(X_train, y_train)

        # Log del modelo de forecast por segmento
        if ml_enabled:
            from mlflow.models import infer_signature
            try:
                signature_enet = infer_signature(X_train, enet.predict(X_train))
            except Exception:
                signature_enet = None
            mlflow.sklearn.log_model(
                sk_model=enet,
                artifact_path=f"segments/{seg_suffix}/model_forecast",
                signature=signature_enet,
                registered_model_name=None
            )
            mlflow.set_tag("model_family", "ENet-forecast")

        # Predicción y residuales
        X_all = scaler.transform(Xy[feat_cols].values.astype(float))
        y_hat_all = enet.predict(X_all)
        df_pred = Xy[["Cliente","Fecha","y_next"]].copy().rename(columns={"y_next":"Volumen_next"})
        df_pred["Volumen_hat"] = y_hat_all
        df_pred["resid"] = df_pred["Volumen_next"] - df_pred["Volumen_hat"]

        # Métricas de forecasting
        rmse = float(np.sqrt(mean_squared_error(y_val, enet.predict(X_val)))) if len(y_val)>0 else np.nan
        mae  = float(mean_absolute_error(y_val, enet.predict(X_val))) if len(y_val)>0 else np.nan
        def _mape_local(y_true, y_pred, eps=1e-8):
            denom = np.maximum(np.abs(y_true), eps)
            return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
        mape_val = _mape_local(y_val, enet.predict(X_val)) if len(y_val)>0 else np.nan
        smape_val = np.mean(200.0 * np.abs(y_val - enet.predict(X_val)) / (np.abs(y_val) + np.abs(enet.predict(X_val)) + 1e-8)) if len(y_val) > 0 else np.nan

        # --- No supervisado en residuales ---
        Ucfg = C.get("unsupervised", {})
        roll_window = int(C.get("residuals", {}).get("roll_window", 30))
        F = residual_features(df_pred, roll_window=roll_window)
        F_train = F.iloc[:cut].copy()
        iso = IsolationForest(
            n_estimators=int(Ucfg.get("n_estimators", 500)),
            max_samples=Ucfg.get("max_samples", "auto"),
            contamination=float(Ucfg.get("contamination", 0.02)),
            random_state=seed,
            n_jobs=-1
        )
        X_if = F_train[["z_abs","resid","r_mean","r_std","dz_abs","dr_abs"]].astype(float).values
        iso.fit(X_if)

        # Log del IF por segmento
        if ml_enabled:
            from mlflow.models import infer_signature
            try:
                signature_if = infer_signature(X_if, iso.decision_function(X_if))
            except Exception:
                signature_if = None
            mlflow.sklearn.log_model(
                sk_model=iso,
                artifact_path=f"segments/{seg_suffix}/model_iforest",
                signature=signature_if,
                registered_model_name=None
            )

        # Scores -> [0,1] y fusión
        X_if_all = F[["z_abs","resid","r_mean","r_std","dz_abs","dr_abs"]].astype(float).values
        dfun = iso.decision_function(X_if_all)        # mayor = normal
        score = -dfun                                  # mayor = anómalo
        smin, smax = float(np.min(score)), float(np.max(score))
        denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
        proba = (score - smin) / denom

        fusion = str(Ucfg.get("fusion","and")).lower()
        z_thr  = float(Ucfg.get("z_threshold", 3.5))
        cont   = float(Ucfg.get("contamination", 0.03))
        thr_if = np.quantile(proba, 1.0 - cont)
        flag_if = (proba >= thr_if).astype(int)
        flag_z  = (F["z_abs"].values >= z_thr).astype(int)
        if fusion == "or":
            Flag_Final = np.where((flag_if + flag_z) > 0, 1, 0)
        else:
            Flag_Final = (flag_if & flag_z).astype(int)

        # === Salidas por segmento ===
        flags = pd.DataFrame({"Cliente": F["Cliente"], "Fecha": F["Fecha"], "proba": proba, "Flag_Final": Flag_Final})
        flags_full = (
            flags.merge(F, on=["Cliente","Fecha"], how="left")
                 .merge(df_seg[["Cliente","Fecha","Volumen","Presion","Temperatura", seg_col]], on=["Cliente","Fecha"], how="left")
                 .merge(df_ssa[["Cliente","Fecha","V_filt"]], on=["Cliente","Fecha"], how="left")
                 .merge(df_pred[["Cliente","Fecha","Volumen_next","Volumen_hat"]], on=["Cliente","Fecha"], how="left")
                 .sort_values(["Cliente","Fecha"]).reset_index(drop=True)
        )

        out = C.get("outputs", {})
        # Sufijos por segmento en artefactos
        base_dir = os.path.dirname(out.get("forecast_model_path","./model_outputs/enet_forecast.pkl")) or "."
        os.makedirs(base_dir, exist_ok=True)

        forecast_pkl = out.get("forecast_model_path","./model_outputs/enet_forecast.pkl")
        scaler_pkl   = out.get("scaler_path","./model_outputs/forecast_scaler.pkl")
        iforest_pkl  = out.get("iforest_path","./model_outputs/iforest_ssa.pkl")
        flags_csv    = out.get("flags_csv","./ctg_anomalias.csv")

        forecast_pkl = forecast_pkl.replace(".pkl", f"__seg={seg_suffix}.pkl")
        scaler_pkl   = scaler_pkl.replace(".pkl", f"__seg={seg_suffix}.pkl")
        iforest_pkl  = iforest_pkl.replace(".pkl", f"__seg={seg_suffix}.pkl")
        flags_csv    = flags_csv.replace(".csv", f"__seg={seg_suffix}.csv")

        with open(forecast_pkl, "wb") as f:
            pickle.dump(enet, f)
        with open(scaler_pkl, "wb") as f:
            pickle.dump(scaler, f)
        with open(iforest_pkl, "wb") as f:
            pickle.dump(iso, f)

        flags_full.to_csv(flags_csv, index=False, encoding="utf-8-sig")

        # Diagnóstico no supervisado
        alarm_rate = float(np.mean(flags["Flag_Final"])) if len(flags) else 0.0
        mean_proba = float(np.mean(flags["proba"])) if len(flags) else 0.0

        # --- MLflow: parámetros y métricas por segmento ---
        if ml_enabled:
            # Usamos prefijos por segmento para evitar colisiones en métricas de múltiples segmentos.
            prefix = f"{seg_suffix}__"

            mlflow.log_params({
                f"{prefix}seed": seed,
                f"{prefix}segment": str(seg),
                f"{prefix}ssa_window": L,
                f"{prefix}ssa_rank": rank,
                f"{prefix}lags_v": lags_v,
                f"{prefix}lags_exo": lags_exo,
                f"{prefix}enet_alpha": float(R.get("alpha", 0.2)),
                f"{prefix}enet_l1_ratio": float(R.get("l1_ratio", 0.1)),
                f"{prefix}test_size": test_size,
                f"{prefix}unsup_n_estimators": int(Ucfg.get("n_estimators", 500)),
                f"{prefix}unsup_contamination": float(Ucfg.get("contamination", 0.02)),
                f"{prefix}unsup_fusion": fusion,
                f"{prefix}unsup_z_threshold": z_thr,
            })
            # Métricas de forecast y no supervisado
            mlflow.log_metric(f"{prefix}rmse_val", rmse if rmse==rmse else 0.0)
            mlflow.log_metric(f"{prefix}mae_val", mae if mae==mae else 0.0)
            mlflow.log_metric(f"{prefix}mape_val", mape_val if mape_val==mape_val else 0.0)
            mlflow.log_metric(f"{prefix}smape_val", smape_val if smape_val==smape_val else 0.0)
            mlflow.log_metric(f"{prefix}alarm_rate", alarm_rate)
            mlflow.log_metric(f"{prefix}mean_proba", mean_proba)

            # Artefactos “crudos” por segmento
            try:
                mlflow.log_artifact(flags_csv, artifact_path=f"segments/{seg_suffix}/outputs")
            except Exception:
                pass

    # Cierra run maestro
    if ml_enabled:
        mlflow.end_run()

    print("OK: entrenamiento por segmento completado. Artefactos y métricas en el run maestro.")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config_ssa.yaml"
    main(cfg)
