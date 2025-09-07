
# -*- coding: utf-8 -*-
"""
Evaluación por segmento compatible con ENet o MLP y (opcional) PolynomialFeatures.
Lee artefactos por segmento:
  model_outputs/enet_forecast__seg=<seg>.pkl        (ENet o MLP)
  model_outputs/forecast_scaler__seg=<seg>.pkl
  model_outputs/iforest_ssa__seg=<seg>.pkl
  model_outputs/poly_features__seg=<seg>.pkl        (opcional)

Genera métricas por segmento y globales (precision/recall/f1/accuracy/roc_auc), además de alarm_rate y thr_if.
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve

try:
    import mlflow
except Exception:
    mlflow = None

# ------------------------------ CONFIG -------------------------------
cfg = {
    "seed": 69069,
    "data": {"segment_col": "Segmento"},
    "paths": {
        "parquet_default": "./df_anom_injection.parquet",
        "enet_model": "./model_outputs/enet_forecast.pkl",
        "scaler": "./model_outputs/forecast_scaler.pkl",
        "iforest": "./model_outputs/iforest_ssa.pkl",
        "poly": "./model_outputs/poly_features.pkl",
        "plots_dir": "./model_outputs/plots",
        "eval_dir": "./model_outputs/eval",
    },
    "ssa": {"window": 24, "rank": 5},
    "regression": {
        "lags_v": 24,
        "lags_exo": 12,
        "poly_v_max_lag": 6,
        "poly_p_max_lag": 4,
        "poly_t_max_lag": 4,
    },
    "residuals": {"roll_window": 12},
    "unsupervised": {"fusion": "or", "z_threshold": 1e9, "contamination": 0.03},
    "mlflow": {
        "enabled": True,
        "tracking_uri": "file:./mlruns",
        "experiment_name": "Contugas-Anomalias",
        "run_name": "eval_injection_by_segment"
    }
}

# ------------------------- Utils -------------------------
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

def _parse_dates_fixed(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["Fecha"] = pd.to_datetime(df["Fecha"].astype(str).str.strip(),
                                     format="%d-%m-%Y %I:%M:%S %p", errors="raise")
    except Exception:
        df["Fecha"] = pd.to_datetime(df["Fecha"].astype(str).str.strip(),
                                     errors="coerce", dayfirst=True)
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

def residual_features(df_pred, roll_window=24):
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

def _find_label_column(df: pd.DataFrame):
    candidates = ["label","labels","is_anom","anom","anomaly","is_anomaly","target","y"]
    for c in candidates:
        if c in df.columns: return c
    known = set(["Cliente","Fecha","Volumen","Presion","Temperatura","V_filt","Volumen_hat",
                 "Volumen_next","resid","z_abs","r_mean","r_std","dz_abs","dr_abs",
                 "proba","Flag_Final","Segmento"])
    for c in df.columns:
        if c in known: continue
        vals = df[c].dropna().unique()
        if len(vals) > 0 and set(np.unique(vals)).issubset({0,1}):
            return c
    raise KeyError("No se encontró columna de etiqueta (p.ej. 'label', 'is_anom').")

def _suffix_from_segment(seg) -> str:
    s = str(seg)
    for ch in r'\/:*?"<>| ':
        s = s.replace(ch, "_")
    return s

def _derive_segment_paths(base_paths: Dict[str,str], seg_suffix: str) -> Dict[str,str]:
    out = {}
    for k in ("enet_model","scaler","iforest","poly"):
        p = base_paths[k]
        root, ext = os.path.splitext(p)
        out[k] = f"{root}__seg={seg_suffix}{ext}"
    return out

# ------------------------- Evaluación por segmento -------------------------
def eval_segment(df_seg: pd.DataFrame, seg_value, C: Dict[str,Any]) -> Tuple[pd.DataFrame, Dict[str,float], str]:
    paths = C["paths"]
    seg_suffix = _suffix_from_segment(seg_value)
    spaths = _derive_segment_paths(paths, seg_suffix)

    # Artefactos
    for key in ("enet_model","scaler","iforest"):
        if not os.path.exists(spaths[key]):
            raise FileNotFoundError(f"[{seg_value}] Falta artefacto: {spaths[key]}")
    with open(spaths["enet_model"], "rb") as f: forecast_model = pickle.load(f)
    with open(spaths["scaler"], "rb") as f: scaler = pickle.load(f)
    with open(spaths["iforest"], "rb") as f: iso = pickle.load(f)

    poly = None
    if os.path.exists(spaths["poly"]):
        try:
            with open(spaths["poly"], "rb") as f:
                poly = pickle.load(f)
        except Exception as e:
            print(f"[WARN] No se cargó poly en {seg_value}: {e}")

    # SSA por cliente
    L    = int(C["ssa"]["window"]); rank = int(C["ssa"]["rank"])
    rows = []
    for cli, g in df_seg.groupby("Cliente"):
        g = g.sort_values("Fecha").copy()
        v = g["Volumen"].astype(float).values
        if len(v) <= L:
            g["V_filt"] = v
        else:
            U,Sv,Vt = ssa_decompose(v, L=L)
            v_rec = ssa_reconstruct_series(U,Sv,Vt, rank=rank)[:len(v)]
            g["V_filt"] = v_rec
        rows.append(g)
    df_ssa = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Tabla supervisada
    R = C["regression"]
    lags_v   = int(R["lags_v"]);  lags_exo = int(R["lags_exo"])
    Xy, feat_cols = make_supervised_table(df_ssa, lags_v=lags_v, lags_exo=lags_exo)

    X_base = Xy[feat_cols].values.astype(float)
    X_concat = X_base

    # Aplicar Poly si existe
    if poly is not None:
        v_max = int(R.get("poly_v_max_lag", 6)); p_max = int(R.get("poly_p_max_lag", 4)); t_max = int(R.get("poly_t_max_lag", 4))
        subs_v = [f"Vf_lag{k}" for k in range(v_max)]
        subs_p = [f"P_lag{k}"  for k in range(p_max)]
        subs_t = [f"T_lag{k}"  for k in range(t_max)]
        subset_cols = [c for c in Xy.columns if c in (subs_v + subs_p + subs_t)]
        try:
            X_sub = Xy[subset_cols].values.astype(float)
            n_in = getattr(poly, "n_input_features_", None)
            if (n_in is not None) and (X_sub.shape[1] != int(n_in)):
                print(f"[WARN] Poly n_input_features_={n_in} != subset={X_sub.shape[1]} en {seg_value}. Omitiendo Poly.")
            else:
                X_poly = poly.transform(X_sub)
                X_concat = np.concatenate([X_base, X_poly], axis=1)
        except Exception as e:
            print(f"[WARN] Fallo Poly en {seg_value}: {e}")

    # Forecast -> residuales
    X_all = scaler.transform(X_concat)
    y_hat = forecast_model.predict(X_all)
    df_pred = Xy[["Cliente","Fecha","y_next"]].copy().rename(columns={"y_next":"Volumen_next"})
    df_pred["Volumen_hat"] = y_hat
    df_pred["resid"] = df_pred["Volumen_next"] - df_pred["Volumen_hat"]

    # Residuales -> features IF
    F = residual_features(df_pred, roll_window=int(C["residuals"]["roll_window"]))

    # Scores IF -> [0,1]
    X_if_all = F[["z_abs","resid","r_mean","r_std","dz_abs","dr_abs"]].astype(float).values
    dfun  = iso.decision_function(X_if_all)           # alto = normal
    score = -dfun                                     # alto = anómalo
    smin, smax = float(np.min(score)), float(np.max(score))
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
    proba = (score - smin) / denom

    fusion = str(C["unsupervised"]["fusion"]).lower()
    z_thr  = float(C["unsupervised"]["z_threshold"])
    cont   = float(C["unsupervised"]["contamination"])
    thr_if = np.quantile(proba, 1.0 - cont)
    flag_if = (proba >= thr_if).astype(int)
    flag_z  = (F["z_abs"].values >= z_thr).astype(int)
    Flag_Final = np.where((flag_if + flag_z) > 0, 1, 0) if fusion == "or" else (flag_if & flag_z).astype(int)

    # y_true alineada a t+1 por cliente
    label_col = _find_label_column(df_seg)
    df_lbl = df_seg[["Cliente","Fecha",label_col]].copy().sort_values(["Cliente","Fecha"])
    df_lbl["label_shiftm1"] = df_lbl.groupby("Cliente")[label_col].shift(-1)

    eval_df = F[["Cliente","Fecha"]].merge(df_lbl[["Cliente","Fecha","label_shiftm1"]],
                                           on=["Cliente","Fecha"], how="left")
    y_true = eval_df["label_shiftm1"].fillna(0).astype(int).values
    y_pred = Flag_Final.astype(int)

    # Métricas
    try:
        roc_auc = roc_auc_score(y_true, proba)
        fpr, tpr, _ = roc_curve(y_true, proba)
    except Exception:
        roc_auc, fpr, tpr = float("nan"), None, None

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    alarm_rate = float(np.mean(Flag_Final)) if len(Flag_Final) else 0.0

    metrics = {
        "segment": str(seg_value),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "positives_true": int(np.sum(y_true)),
        "positives_pred": int(np.sum(y_pred)),
        "n_eval": int(len(y_true)),
        "alarm_rate": float(alarm_rate),
        "thr_if": float(thr_if),
        "contamination": float(cont),
        "fusion": fusion,
        "z_threshold": float(z_thr),
    }

    os.makedirs(C["paths"]["plots_dir"], exist_ok=True)
    roc_path = None
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"ROC {seg_value} (AUC={roc_auc:.3f})")
        plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"Curva ROC - Inyección ({seg_value})")
        plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
        roc_path = os.path.join(C["paths"]["plots_dir"], f"roc_injection__seg={_suffix_from_segment(seg_value)}.png")
        plt.savefig(roc_path, dpi=140); plt.close()

    out_df = F.copy()
    out_df["proba"] = proba
    out_df["Flag_Final"] = Flag_Final
    out_df = out_df.merge(df_lbl[["Cliente","Fecha","label_shiftm1"]], on=["Cliente","Fecha"], how="left") \
                   .rename(columns={"label_shiftm1":"label_next"})
    out_df = out_df.merge(df_seg[["Cliente","Fecha","Volumen","Presion","Temperatura"]], on=["Cliente","Fecha"], how="left")
    out_df = out_df.merge(df_pred[["Cliente","Fecha","Volumen_next","Volumen_hat"]], on=["Cliente","Fecha"], how="left")
    out_df["Segmento"] = seg_value

    return out_df, metrics, roc_path

# ------------------------- Main -------------------------
def main(parquet_path=None):
    global mlflow
    seed = int(cfg.get("seed", 69069))
    np.random.seed(seed)

    paths = cfg["paths"]
    parquet_path = parquet_path or paths["parquet_default"]
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No encuentro el parquet: {parquet_path}")

    mlconf = cfg.get("mlflow", {})
    ml_enabled = bool(mlconf.get("enabled", False) and (mlflow is not None))
    if ml_enabled:
        mlflow.set_tracking_uri(mlconf.get("tracking_uri"))
        mlflow.set_experiment(mlconf.get("experiment_name", "Contugas-Anomalias"))
        mlflow.start_run(run_name=mlconf.get("run_name", "eval_injection_by_segment"))
        mlflow.set_tag("phase", "evaluation")
        mlflow.set_tag("eval_type", "injection_parquet_by_segment")

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = _normalize_columns(df); df = _parse_dates_fixed(df)

    req = ["Cliente","Fecha","Volumen","Presion","Temperatura"]
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Faltan columnas requeridas: {miss}")
    for c in ["Volumen","Presion","Temperatura"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=req).sort_values(["Cliente","Fecha"]).reset_index(drop=True)

    seg_col = cfg.get("data",{}).get("segment_col","Segmento")
    if seg_col not in df.columns:
        raise KeyError(f"No encuentro la columna de segmento '{seg_col}'.")

    segments = [s for s in df[seg_col].dropna().unique().tolist()]
    if len(segments) == 0:
        raise ValueError(f"La columna '{seg_col}' no tiene valores válidos.")

    if ml_enabled:
        try:
            import mlflow.data
            ds = mlflow.data.from_pandas(df, source=os.path.abspath(parquet_path), name="contugas_injection_eval_by_segment")
            mlflow.log_input(ds, context="validation")
        except Exception:
            mlflow.set_tag("eval_dataset_path", os.path.abspath(parquet_path))

    os.makedirs(paths["eval_dir"], exist_ok=True)
    all_preds, metrics_list = [], []

    for seg in segments:
        df_seg = df[df[seg_col] == seg].copy()
        if df_seg.empty: continue

        if ml_enabled:
            mlflow.start_run(run_name=f"eval_segment={_suffix_from_segment(seg)}", nested=True)
            mlflow.set_tag("segment", str(seg))

        try:
            preds_df, metrics, roc_path = eval_segment(df_seg, seg, cfg)
            all_preds.append(preds_df); metrics_list.append(metrics)

            seg_sfx = _suffix_from_segment(seg)
            out_csv = os.path.join(paths["eval_dir"], f"preds_injection__seg={seg_sfx}.csv")
            preds_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            with open(os.path.join(paths["eval_dir"], f"metrics_injection__seg={seg_sfx}.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            if ml_enabled:
                mlflow.log_metrics({
                    "precision": metrics["precision"], "recall": metrics["recall"],
                    "f1": metrics["f1"], "accuracy": metrics["accuracy"],
                    "roc_auc": 0.0 if np.isnan(metrics["roc_auc"]) else metrics["roc_auc"],
                    "positives_true": metrics["positives_true"],
                    "positives_pred": metrics["positives_pred"],
                    "n_eval": metrics["n_eval"],
                    "alarm_rate": metrics["alarm_rate"],
                    "thr_if": metrics["thr_if"],
                })
                mlflow.log_params({
                    "fusion": metrics.get("fusion",""), "z_threshold": metrics.get("z_threshold", 0.0),
                    "contamination": metrics.get("contamination", 0.0),
                })
                if roc_path and os.path.exists(roc_path):
                    mlflow.log_artifact(roc_path, artifact_path=f"segments/{seg_sfx}/plots")
                mlflow.log_artifact(out_csv, artifact_path=f"segments/{seg_sfx}/eval")
                mlflow.log_artifact(os.path.join(paths["eval_dir"], f"metrics_injection__seg={seg_sfx}.json"),
                                    artifact_path=f"segments/{seg_sfx}/eval")
        finally:
            if ml_enabled: mlflow.end_run()

    if len(all_preds) > 0:
        preds_all = pd.concat(all_preds, ignore_index=True)
        y_true_all = preds_all["label_next"].fillna(0).astype(int).values
        y_pred_all = preds_all["Flag_Final"].astype(int).values
        proba_all  = preds_all["proba"].astype(float).values

        try:
            roc_auc_all = roc_auc_score(y_true_all, proba_all)
            fpr_all, tpr_all, _ = roc_curve(y_true_all, proba_all)
        except Exception:
            roc_auc_all, fpr_all, tpr_all = float("nan"), None, None

        metrics_all = {
            "precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
            "recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
            "f1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "roc_auc": float(roc_auc_all),
            "positives_true": int(np.sum(y_true_all)),
            "positives_pred": int(np.sum(y_pred_all)),
            "n_eval": int(len(y_true_all)),
            "alarm_rate": float(np.mean(y_pred_all)) if len(y_pred_all) else 0.0,
        }

        preds_all_csv = os.path.join(paths["eval_dir"], "preds_injection__all_segments.csv")
        preds_all.to_csv(preds_all_csv, index=False, encoding="utf-8-sig")
        with open(os.path.join(paths["eval_dir"], "metrics_injection__all_segments.json"), "w", encoding="utf-8") as f:
            json.dump({"overall": metrics_all, "by_segment": metrics_list}, f, indent=2, ensure_ascii=False)

        roc_all_path = None
        if fpr_all is not None and tpr_all is not None:
            plt.figure(figsize=(6,5))
            plt.plot(fpr_all, tpr_all, label=f"ROC ALL (AUC={roc_auc_all:.3f})")
            plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
            plt.title("Curva ROC - Inyección (Todos los segmentos)")
            plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
            os.makedirs(cfg["paths"]["plots_dir"], exist_ok=True)
            roc_all_path = os.path.join(cfg["paths"]["plots_dir"], "roc_injection__all_segments.png")
            plt.savefig(roc_all_path, dpi=140); plt.close()

        if ml_enabled:
            mlflow.log_metrics({
                "precision_overall": metrics_all["precision"],
                "recall_overall": metrics_all["recall"],
                "f1_overall": metrics_all["f1"],
                "accuracy_overall": metrics_all["accuracy"],
                "roc_auc_overall": 0.0 if np.isnan(metrics_all["roc_auc"]) else metrics_all["roc_auc"],
                "positives_true_overall": metrics_all["positives_true"],
                "positives_pred_overall": metrics_all["positives_pred"],
                "n_eval_overall": metrics_all["n_eval"],
                "alarm_rate_overall": metrics_all["alarm_rate"],
            })
            mlflow.log_artifact(preds_all_csv, artifact_path="eval/all_segments")
            mlflow.log_artifact(os.path.join(paths["eval_dir"], "metrics_injection__all_segments.json"),
                                artifact_path="eval/all_segments")
            if roc_all_path and os.path.exists(roc_all_path):
                mlflow.log_artifact(roc_all_path, artifact_path="plots")

    if ml_enabled:
        mlflow.end_run()
    print("OK: evaluación por segmento completada.")

if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else None
    main(parquet)
