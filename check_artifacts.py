# check_artifacts.py
import mlflow
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("🔍 Verificando estructura de artifacts...\n")

client = mlflow.tracking.MlflowClient()

# Obtener experimento
experiment = client.get_experiment_by_name("MuniRisk-Bolivia")
if experiment:
    runs = client.search_runs(experiment.experiment_id)
    
    for run in runs:
        print(f"Run: {run.info.run_name}")
        print(f"  ID: {run.info.run_id}")
        
        # Listar todos los artifacts
        artifacts = client.list_artifacts(run.info.run_id)
        print(f"  Artifacts encontrados:")
        for artifact in artifacts:
            print(f"    - {artifact.path} ({artifact.is_dir})")
            
            # Si es directorio, listar contenido
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run.info.run_id, artifact.path)
                for sub in sub_artifacts:
                    print(f"      - {sub.path}")
        print()