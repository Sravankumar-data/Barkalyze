# barkalyze

# Workflows
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

uvicorn api:app --reload
# uvicorn connect.api:app


docker-compose down --volumes
docker-compose up --build


FastAPI backend at http://localhost:8000

Prometheus at http://localhost:9090

Grafana at http://localhost:3000

need to add in Grafana http://prometheus:9090

dvc init

set PYTHONPATH=.
dvc repro

dvc dag

# cleaning dataset

dvc remove artifacts/data_ingestion/emotion_dataset.dvc

dvc gc -w --force

git add .
git commit -m "Remove old dataset for reset"
git push origin main

dvc push --force

# Add Your Cleaned v1 Dataset

dvc add data_versioning/emotion_dataset

git add data_versioning/emotion_dataset.dvc .gitignore
git commit -m "Add emotion dataset version 1"

git tag v1
git push origin v1

