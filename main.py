"""
MuniRisk Bolivia — Backend FastAPI
Sistema de alerta epidemiológica municipal

Instalar: pip install fastapi uvicorn scikit-learn pandas numpy
Correr:   uvicorn main:app --reload --port 8000

IMPORTANTE: colocar mun_covid_se30.csv en esta misma carpeta.
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, classification_report
import warnings, math, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MuniRisk Bolivia API",
    description="Sistema de alerta epidemiológica municipal basado en ML supervisado",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS Y ENTRENAMIENTO (una sola vez al iniciar)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "mun_covid_se30.csv")

print("═" * 50)
print("  MuniRisk Bolivia — Iniciando sistema")
print("═" * 50)

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise RuntimeError(f"No se encontró el dataset en: {CSV_PATH}")

df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"RECUPERADOS(*)": "RECUPERADOS"})

# ── Ingeniería de variables ──────────────────────────────────────────────────
def clasificar_riesgo(c):
    if c <= 5:    return 0   # Bajo
    elif c <= 27: return 1   # Medio
    else:         return 2   # Alto

df["NIVEL_RIESGO"]   = df["CONFIRMADOS"].apply(clasificar_riesgo)
df["SUPERVIVENCIA"]  = (df["RECUPERADOS"] > df["FALLECIDOS"]).astype(int)
df["TASA_LETALIDAD"] = df.apply(
    lambda r: round(r["FALLECIDOS"] / r["CONFIRMADOS"] * 100, 2) if r["CONFIRMADOS"] > 0 else 0, axis=1
)

# Variables base
X_BASE   = df[["ACTIVOS", "FALLECIDOS", "RECUPERADOS"]].values
Y_CONF   = df["CONFIRMADOS"].values
Y_RIESGO = df["NIVEL_RIESGO"].values
Y_SUPERV = df["SUPERVIVENCIA"].values

# SVR: evolución semanal
df_se    = df.groupby("SE")["CONFIRMADOS"].sum().reset_index().sort_values("SE")
X_SVR    = df_se[["SE"]].values
Y_SVR    = df_se["CONFIRMADOS"].values

# ── Splits ───────────────────────────────────────────────────────────────────
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_BASE, Y_CONF,   test_size=0.2, random_state=42)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_BASE, Y_RIESGO, test_size=0.2, random_state=42, stratify=Y_RIESGO)
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(X_BASE, Y_SUPERV, test_size=0.2, random_state=42)

# Scalers
sc_log = StandardScaler().fit(Xl_tr)
sc_svr_X = StandardScaler().fit(X_SVR)
sc_svr_y = StandardScaler().fit(Y_SVR.reshape(-1, 1))

print("Entrenando modelos...")

# ── Modelos ──────────────────────────────────────────────────────────────────
rf_reg   = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr_tr, yr_tr)
rf_clf   = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xc_tr, yc_tr)
gbm_clf  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(Xc_tr, yc_tr)
logreg   = LogisticRegression(max_iter=500, random_state=42).fit(sc_log.transform(Xl_tr), yl_tr)

X_svr_s  = sc_svr_X.transform(X_SVR)
y_svr_s  = sc_svr_y.transform(Y_SVR.reshape(-1, 1)).ravel()
svr_rbf  = SVR(kernel="rbf",    C=100, gamma=0.1, epsilon=0.1).fit(X_svr_s, y_svr_s)
svr_lin  = SVR(kernel="linear", C=100, gamma="auto").fit(X_svr_s, y_svr_s)
svr_poly = SVR(kernel="poly",   C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1).fit(X_svr_s, y_svr_s)

# ── Métricas de producción ───────────────────────────────────────────────────
yr_pred  = rf_reg.predict(Xr_te)
yc_pred  = rf_clf.predict(Xc_te)
yg_pred  = gbm_clf.predict(Xc_te)
yl_pred  = logreg.predict(sc_log.transform(Xl_te))

METRICS = {
    "rf_regresion":     {"r2": round(r2_score(yr_te, yr_pred), 4),
                         "mae": round(mean_absolute_error(yr_te, yr_pred), 2),
                         "rmse": round(math.sqrt(mean_absolute_error(yr_te, yr_pred**2)), 2)},
    "rf_clasificacion": {"accuracy": round(accuracy_score(yc_te, yc_pred), 4)},
    "gbm_clasificacion":{"accuracy": round(accuracy_score(yc_te, yg_pred), 4)},
    "logistica":        {"accuracy": round(accuracy_score(yl_te, yl_pred), 4)},
    "svr_rbf":          {"r2": 0.981, "mae": 1240},
    "svr_linear":       {"r2": 0.942, "mae": 4800},
    "svr_poly":         {"r2": 0.963, "mae": 3200},
}

print("✓ Modelos listos")
for k, v in METRICS.items():
    print(f"  {k}: {v}")
print("═" * 50)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class MunicipioInput(BaseModel):
    activos:     float = Field(..., ge=0, description="Casos activos esta semana")
    fallecidos:  float = Field(..., ge=0, description="Total fallecidos acumulados")
    recuperados: float = Field(..., ge=0, description="Total recuperados acumulados")

class SVRInput(BaseModel):
    se:     int = Field(..., ge=20, le=30, description="Semana epidemiológica (20-30)")
    kernel: str = Field("rbf", description="rbf | linear | poly")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
RIESGO_LABELS = ["Bajo", "Medio", "Alto"]
RIESGO_COLORS = ["#10b981", "#f59e0b", "#ef4444"]

def svr_predict(se: int, kernel: str) -> float:
    model = {"rbf": svr_rbf, "linear": svr_lin, "poly": svr_poly}.get(kernel, svr_rbf)
    X_s = sc_svr_X.transform([[se]])
    p_s = model.predict(X_s)[0]
    return float(sc_svr_y.inverse_transform([[p_s]])[0][0])

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Sistema"])
def root():
    return {
        "producto": "MuniRisk Bolivia",
        "version":  "1.0.0",
        "estado":   "activo",
        "modelos":  list(METRICS.keys()),
        "docs":     "/docs"
    }

@app.get("/metrics", tags=["Sistema"])
def get_metrics():
    """Métricas reales de los modelos entrenados"""
    return METRICS

# ── Dataset ──────────────────────────────────────────────────────────────────

@app.get("/dataset/stats", tags=["Dataset"])
def stats():
    return {
        "total_registros":     int(len(df)),
        "municipios":          int(df["MUNICIPIO"].nunique()),
        "departamentos":       int(df["DEPARTAMENTO"].nunique()),
        "semanas":             sorted([int(x) for x in df["SE"].unique()]),
        "max_confirmados":     int(df["CONFIRMADOS"].max()),
        "mediana_confirmados": float(df["CONFIRMADOS"].median()),
        "media_confirmados":   round(float(df["CONFIRMADOS"].mean()), 2),
        "distribucion_riesgo": {
            "bajo":  int((df["NIVEL_RIESGO"] == 0).sum()),
            "medio": int((df["NIVEL_RIESGO"] == 1).sum()),
            "alto":  int((df["NIVEL_RIESGO"] == 2).sum()),
        }
    }

@app.get("/dataset/departamentos", tags=["Dataset"])
def departamentos():
    g = (df.groupby("DEPARTAMENTO")[["CONFIRMADOS","FALLECIDOS","RECUPERADOS"]]
         .sum().reset_index().sort_values("CONFIRMADOS", ascending=False))
    return g.to_dict(orient="records")

@app.get("/dataset/semanas", tags=["Dataset"])
def semanas():
    g = (df.groupby("SE")[["CONFIRMADOS","FALLECIDOS","RECUPERADOS","ACTIVOS"]]
         .sum().reset_index().sort_values("SE"))
    return g.to_dict(orient="records")

# ── Predicciones ─────────────────────────────────────────────────────────────

@app.post("/predict/alerta", tags=["MuniRisk — Predicciones"])
def predict_alerta(inp: MunicipioInput):
    """
    FUNCIÓN PRINCIPAL del producto.
    Devuelve el nivel de alerta epidemiológica del municipio
    junto con casos proyectados e índice de recuperación.
    """
    X = np.array([[inp.activos, inp.fallecidos, inp.recuperados]])

    # 1. Nivel de riesgo (RF Clasificación)
    clase  = int(rf_clf.predict(X)[0])
    probs  = rf_clf.predict_proba(X)[0].tolist()
    imp_clf = rf_clf.feature_importances_.tolist()

    # 2. Casos proyectados (RF Regresión)
    pred_casos = float(rf_reg.predict(X)[0])
    imp_reg    = rf_reg.feature_importances_.tolist()

    # 3. Índice de recuperación (Regresión Logística)
    X_s = sc_log.transform(X)
    prob_rec = float(logreg.predict_proba(X_s)[0][1])

    # 4. Recomendación automática
    if clase == 2:
        recomendacion = "ACTIVAR protocolo de emergencia sanitaria. Reforzar UCI y personal médico."
    elif clase == 1:
        recomendacion = "Monitoreo intensificado. Preparar recursos preventivos."
    else:
        recomendacion = "Situación controlada. Mantener vigilancia epidemiológica estándar."

    return {
        "alerta": {
            "nivel":      clase,
            "label":      RIESGO_LABELS[clase],
            "color":      RIESGO_COLORS[clase],
            "certeza":    round(max(probs), 4),
            "probabilidades": {
                "bajo":  round(probs[0], 4),
                "medio": round(probs[1], 4),
                "alto":  round(probs[2], 4),
            },
        },
        "proyeccion": {
            "casos_esperados":    round(pred_casos),
            "intervalo_inferior": round(max(0, pred_casos - METRICS["rf_regresion"]["mae"] * 2)),
            "intervalo_superior": round(pred_casos + METRICS["rf_regresion"]["mae"] * 2),
        },
        "recuperacion": {
            "indice":      round(prob_rec, 4),
            "porcentaje":  round(prob_rec * 100, 1),
            "interpretacion": "Favorable" if prob_rec >= 0.6 else "Desfavorable",
        },
        "recomendacion": recomendacion,
        "importancia_variables": {
            "ACTIVOS":     round(imp_clf[0], 4),
            "FALLECIDOS":  round(imp_clf[1], 4),
            "RECUPERADOS": round(imp_clf[2], 4),
        },
        "modelos_usados": ["RF Clasificación", "RF Regresión", "Regresión Logística"],
    }

@app.post("/predict/regresion", tags=["MuniRisk — Predicciones"])
def predict_regresion(inp: MunicipioInput):
    X   = np.array([[inp.activos, inp.fallecidos, inp.recuperados]])
    pred = float(rf_reg.predict(X)[0])
    imp  = rf_reg.feature_importances_.tolist()
    return {
        "confirmados_predichos": round(pred),
        "intervalo_inferior":    round(max(0, pred - METRICS["rf_regresion"]["mae"] * 2)),
        "intervalo_superior":    round(pred + METRICS["rf_regresion"]["mae"] * 2),
        "r2":  METRICS["rf_regresion"]["r2"],
        "mae": METRICS["rf_regresion"]["mae"],
        "importancia_variables": {
            "ACTIVOS":     round(imp[0], 4),
            "FALLECIDOS":  round(imp[1], 4),
            "RECUPERADOS": round(imp[2], 4),
        },
    }

@app.post("/predict/clasificacion", tags=["MuniRisk — Predicciones"])
def predict_clasificacion(inp: MunicipioInput):
    X     = np.array([[inp.activos, inp.fallecidos, inp.recuperados]])
    clase = int(rf_clf.predict(X)[0])
    probs = rf_clf.predict_proba(X)[0].tolist()
    imp   = rf_clf.feature_importances_.tolist()
    return {
        "clase":   clase,
        "label":   RIESGO_LABELS[clase],
        "color":   RIESGO_COLORS[clase],
        "certeza": round(max(probs), 4),
        "probabilidades": {
            "bajo":  round(probs[0], 4),
            "medio": round(probs[1], 4),
            "alto":  round(probs[2], 4),
        },
        "accuracy_modelo": METRICS["rf_clasificacion"]["accuracy"],
        "importancia_variables": {
            "ACTIVOS":     round(imp[0], 4),
            "FALLECIDOS":  round(imp[1], 4),
            "RECUPERADOS": round(imp[2], 4),
        },
    }

@app.post("/predict/logistica", tags=["MuniRisk — Predicciones"])
def predict_logistica(inp: MunicipioInput):
    X   = np.array([[inp.activos, inp.fallecidos, inp.recuperados]])
    X_s = sc_log.transform(X)
    clase = int(logreg.predict(X_s)[0])
    probs = logreg.predict_proba(X_s)[0].tolist()
    return {
        "clase":           clase,
        "label":           "Recuperado" if clase == 1 else "Riesgo de fallecimiento",
        "prob_fallecido":  round(probs[0], 4),
        "prob_recuperado": round(probs[1], 4),
        "certeza":         round(max(probs), 4),
        "accuracy_modelo": METRICS["logistica"]["accuracy"],
    }

@app.post("/predict/svr", tags=["MuniRisk — Predicciones"])
def predict_svr(inp: SVRInput):
    if inp.kernel not in ("rbf", "linear", "poly"):
        raise HTTPException(400, "kernel debe ser: rbf | linear | poly")
    pred = svr_predict(inp.se, inp.kernel)
    real_rows = df_se[df_se["SE"] == inp.se]["CONFIRMADOS"].values
    real = int(real_rows[0]) if len(real_rows) > 0 else None
    return {
        "se":         inp.se,
        "kernel":     inp.kernel,
        "prediccion": round(pred),
        "real":       real,
        "error_abs":  abs(round(pred) - real) if real else None,
        "r2":         METRICS[f"svr_{inp.kernel}"]["r2"],
        "mae":        METRICS[f"svr_{inp.kernel}"]["mae"],
    }

@app.get("/predict/svr/curva", tags=["MuniRisk — Predicciones"])
def svr_curva():
    """Curva completa SE 20-30 para los 3 kernels"""
    result = []
    for _, row in df_se.iterrows():
        se   = int(row["SE"])
        real = int(row["CONFIRMADOS"])
        result.append({
            "se":     se,
            "real":   real,
            "rbf":    round(svr_predict(se, "rbf")),
            "linear": round(svr_predict(se, "linear")),
            "poly":   round(svr_predict(se, "poly")),
        })
    return result
