"""
mlops.py — Entrenamiento con MLflow tracking (versión SQLite)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import json
import os

# ── Configurar MLflow con SQLite ─────────────────────────────────────────────
# Eliminar configuración anterior si existe
if os.path.exists("./mlruns"):
    print("⚠️  Se detectó configuración anterior con FileStore")
    print("   Se usará SQLite para mejor compatibilidad")

# Usar SQLite en su lugar
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MuniRisk-Bolivia")

# ── Cargar datos ─────────────────────────────────────────────────────────────
df = pd.read_csv("mun_covid_se30.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"RECUPERADOS(*)": "RECUPERADOS"})

def clasificar(c):
    if c <= 5:    return 0
    elif c <= 27: return 1
    else:         return 2

df["NIVEL_RIESGO"] = df["CONFIRMADOS"].apply(clasificar)
X = df[["ACTIVOS","FALLECIDOS","RECUPERADOS"]].values
y_reg  = df["CONFIRMADOS"].values
y_clf  = df["NIVEL_RIESGO"].values

Xr_tr,Xr_te,yr_tr,yr_te = train_test_split(X,y_reg, test_size=0.2,random_state=42)
Xc_tr,Xc_te,yc_tr,yc_te = train_test_split(X,y_clf, test_size=0.2,random_state=42,stratify=y_clf)

# Variables para almacenar métricas
metrics_data = {}

# ── Experimento v1 ───────────────────────────────────────────────────────────
with mlflow.start_run(run_name="RF_Clasif_v1"):
    params = {"n_estimators": 100, "max_depth": None, "random_state": 42,
              "umbral_bajo": 5, "umbral_alto": 27}
    mlflow.log_params(params)

    model = RandomForestClassifier(**{k:v for k,v in params.items()
                                      if k in ["n_estimators","max_depth","random_state"]})
    model.fit(Xc_tr, yc_tr)
    acc = accuracy_score(yc_te, model.predict(Xc_te))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("umbral_bajo", params["umbral_bajo"])
    mlflow.log_metric("umbral_alto", params["umbral_alto"])
    mlflow.sklearn.log_model(model, "RF_Clasificacion_v1")
    
    print(f"v1 — Accuracy: {acc:.4f}")
    metrics_data["rf_clasificacion_v1"] = {"accuracy": acc}

# ── Experimento v2 (más sensible) ────────────────────────────────────────────
def clasificar_v2(c):
    if c <= 3:    return 0
    elif c <= 15: return 1
    else:         return 2

df["NIVEL_V2"] = df["CONFIRMADOS"].apply(clasificar_v2)
yc2 = df["NIVEL_V2"].values
Xc2_tr,Xc2_te,yc2_tr,yc2_te = train_test_split(X,yc2,test_size=0.2,random_state=42,stratify=yc2)

with mlflow.start_run(run_name="RF_Clasif_v2"):
    params2 = {"n_estimators": 150, "max_depth": None, "random_state": 42,
               "umbral_bajo": 3, "umbral_alto": 15}
    mlflow.log_params(params2)

    model2 = RandomForestClassifier(n_estimators=150, random_state=42)
    model2.fit(Xc2_tr, yc2_tr)
    acc2 = accuracy_score(yc2_te, model2.predict(Xc2_te))

    mlflow.log_metric("accuracy", acc2)
    mlflow.log_metric("umbral_bajo", params2["umbral_bajo"])
    mlflow.log_metric("umbral_alto", params2["umbral_alto"])
    mlflow.sklearn.log_model(model2, "RF_Clasificacion_v2")
    
    print(f"v2 — Accuracy: {acc2:.4f}")
    metrics_data["rf_clasificacion_v2"] = {"accuracy": acc2}

# ── RF Regresión ──────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="RF_Regresion_v1"):
    mlflow.log_params({"n_estimators": 100, "random_state": 42})
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr_tr, yr_tr)
    pred = reg.predict(Xr_te)
    r2  = r2_score(yr_te, pred)
    mae = mean_absolute_error(yr_te, pred)
    
    mlflow.log_metric("r2",  r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(reg, "RF_Regresion_v1")
    
    print(f"RF Reg — R²: {r2:.4f} | MAE: {mae:.2f}")
    metrics_data["rf_regresion"] = {"r2": r2, "mae": mae}

# Guardar métricas para DVC
with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)

print("\n✅ Experimentos registrados con SQLite")
print("📊 Para ver la interfaz: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("🌐 Abrí: http://localhost:5000")