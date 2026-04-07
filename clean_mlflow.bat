@echo off
echo Limpiando configuración anterior de MLflow...
if exist mlruns rmdir /s /q mlruns
if exist mlflow.db del mlflow.db
if exist mlflow.db-journal del mlflow.db-journal
echo.
echo Configuración limpiada. Ahora ejecuta: python mlops.py