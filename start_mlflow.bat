@echo off
echo Iniciando MLflow con SQLite...
echo.
echo Para ver la interfaz web:
echo mlflow ui --backend-store-uri sqlite:///mlflow.db
echo.
echo Luego abre http://localhost:5000
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000