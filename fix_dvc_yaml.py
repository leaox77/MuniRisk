# fix_dvc_yaml.py
import os

dvc_content = """stages:
  entrenar:
    cmd: python mlops.py
    deps:
      - mlops.py
      - mun_covid_se30.csv
    metrics:
      - metrics.json:
          cache: false
    outs:
      - modelos/
"""

with open("dvc.yaml", "w") as f:
    f.write(dvc_content)

print("✅ dvc.yaml corregido")
print("\nContenido:")
print(dvc_content)
print("\nAhora ejecuta: dvc add mun_covid_se30.csv")