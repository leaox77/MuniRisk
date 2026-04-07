#!/bin/bash
# Ejecutar una sola vez para inicializar DVC

git init                          # si no tenés git ya
dvc init
dvc add mun_covid_se30.csv        # versiona el CSV
git add mun_covid_se30.csv.dvc .dvcignore
git commit -m "feat: agregar dataset COVID Bolivia con DVC"

echo "DVC inicializado. El CSV está versionado."
echo "Para reproducir el pipeline: dvc repro"