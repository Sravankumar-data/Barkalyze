# BARKALYZE

_Transforming Data into Actionable Insights Instantly_

![Last Commit](https://img.shields.io/badge/last%20commit-today-brightgreen)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-91.2%25-blue)
![Languages](https://img.shields.io/badge/languages-9-blue)

---

## 🚀 Built with the tools and technologies:

| Tech Stack |
|------------|
| ![JSON](https://img.shields.io/badge/-JSON-black?logo=json&logoColor=white) ![Markdown](https://img.shields.io/badge/-Markdown-black?logo=markdown&logoColor=white) ![Streamlit](https://img.shields.io/badge/-Streamlit-ff4b4b?logo=streamlit&logoColor=white) ![npm](https://img.shields.io/badge/-npm-cb3837?logo=npm&logoColor=white) ![Prometheus](https://img.shields.io/badge/-Prometheus-e6522c?logo=prometheus&logoColor=white) ![Grafana](https://img.shields.io/badge/-Grafana-f46800?logo=grafana&logoColor=white) ![TensorFlow](https://img.shields.io/badge/-TensorFlow-ff6f00?logo=tensorflow&logoColor=white) ![Prettier](https://img.shields.io/badge/-Prettier-f7b93e?logo=prettier&logoColor=black) |
| ![JavaScript](https://img.shields.io/badge/-JavaScript-f7df1e?logo=javascript&logoColor=black) ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white) ![DVC](https://img.shields.io/badge/-DVC-13ADC7?logo=dvc&logoColor=white) |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) ![Webpack](https://img.shields.io/badge/-Webpack-1C78C0?logo=webpack&logoColor=white) ![MLflow](https://img.shields.io/badge/-MLflow-0174BF?logo=mlflow&logoColor=white) ![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white) ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) ![TypeScript](https://img.shields.io/badge/-TypeScript-3178C6?logo=typescript&logoColor=white) ![Lodash](https://img.shields.io/badge/-Lodash-3492FF?logo=lodash&logoColor=white) ![bat](https://img.shields.io/badge/-bat-5C2D91?logo=visualstudiocode&logoColor=white) ![pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white) |
| ![YAML](https://img.shields.io/badge/-YAML-red?logo=yaml&logoColor=white) |

---


> ⚡ A project made by developers, for developers — focused on showcasing **AI engineering and MLOps** skills, not solving a real-world problem.

# Pipeline Workflows
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


## 📘 Project Overview

**Barkalyze** is a real-time emotion analysis and visualization platform that:

- Captures webcam input using `Streamlit WebRTC`
- Predicts emotions using a TensorFlow Lite model
- Automatically plays emotion-matching dog videos
- Tracks metrics using `Prometheus` and visualizes with `Grafana`
- Supports experiment tracking via `MLflow`
- Organizes datasets using `DVC`
- Interacts with `MongoDB` and `GridFS` for storage and streaming data

---

## 🧠 Key Features

- 🖥️ Live Emotion Detection
- 🐶 Smart Video Playback Based on Mood
- 📊 Prometheus Monitoring + Grafana Dashboards
- 📁 Cloud-Based Dataset & Metadata Management
- ⚙️ FastAPI Backend with REST Endpoints

---

## 📂 Project Structure

```bash
├── Backend/
│   ├── main.py (FastAPI)
│   └── metrics.py (Prometheus Exporters)
├── Frontend/
│   └── streamlit_app.py
├── notebooks/
│   └── emotion_training.ipynb
├── dvc.yaml
├── Dockerfile
└── README.md


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

