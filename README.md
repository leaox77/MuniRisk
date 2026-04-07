# 1. Si cambias el CSV (nuevos datos)
dvc add mun_covid_se30.csv  # Actualizar hash del dataset
git add mun_covid_se30.csv.dvc
git commit -m "Actualizar dataset"

# 2. Ejecutar pipeline actualizado
dvc repro

# 3. Ver cambios en métricas
dvc metrics show --all  # Comparar con versiones anteriores

# Ver todas las métricas históricas
dvc metrics show --all

# Comparar dos versiones
dvc metrics show --all --md  # Formato markdown

# Ver qué está trackeando DVC
dvc list . --dvc-only

# Ver diferencias entre versiones
dvc diff

# Ver pipeline visualmente
dvc dag

# Ejecutar sin cache (re-entrenar todo)
dvc repro --force

# Ver métricas en formato tabla
dvc metrics show --all --md

# Limpiar cache de DVC (si necesitas espacio)
dvc gc

# 1. Entrenar con MLflow (tracking de experimentos)
python mlops.py

# 2. Versionar datos con DVC
dvc add mun_covid_se30.csv

# 3. Pipeline reproducible
dvc repro

# 4. Ver UI de MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db