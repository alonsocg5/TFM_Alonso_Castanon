# Sistema de Apoyo a la Decisión Clínica en Oncología Pediátrica

**Trabajo Fin de Máster — Big Data, Data Science & IA**  
Universidad Complutense de Madrid | Curso 2025–2026  
**Autor:** Alonso Castañón González

---

## Descripción

Sistema que integra dos capas tecnológicas para apoyar la decisión clínica en oncología pediátrica:

- **Capa ML:** modelo predictivo de supervivencia a 5 años (XGBoost + SHAP) para osteosarcoma y sarcoma de Ewing pediátrico
- **Capa RAG:** asistente en lenguaje natural (LangChain + Qdrant) para oncólogos y familias de pacientes

---

## Estructura del proyecto

```
TFM_Alonso_Castanon/
│
├── data/                          # Dataset SEER (no incluido, ver abajo)
├── notebooks/
│   ├── 01_EDA_limpieza.ipynb      # Análisis exploratorio y limpieza
│   ├── 02_modelo_ML.ipynb         # Entrenamiento, evaluación y SHAP
│   └── 03_RAG_pipeline.ipynb      # Pipeline RAG con LangChain y Qdrant
├── src/
│   ├── api.py                     # API REST con FastAPI
│   └── rag.py                     # Pipeline RAG
├── models/                        # Modelo serializado (joblib)
├── memoria/                       # Memoria del TFM (PDF final)
└── requirements.txt
```

---

## Datos

Los datos provienen de la base de datos pública **SEER (Surveillance, Epidemiology, and End Results)** del National Cancer Institute (NCI).

Para reproducir el proyecto:
1. Solicitar acceso en [seer.cancer.gov](https://seer.cancer.gov/data/access.html)
2. Descargar el dataset: `Incidence - SEER Research Data, 17 Registries, Nov 2025 Sub (2000-2023)`
3. Filtrar por histología 9180-9187 (osteosarcoma) y 9260 (sarcoma de Ewing), edad 0-19 años
4. Colocar el CSV exportado en `data/seer_osteosarcoma_ewing.csv`

> Los datos no se incluyen en este repositorio por el Data Use Agreement firmado con el NCI.

---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/TFM_Alonso_Castanon.git
cd TFM_Alonso_Castanon
pip install -r requirements.txt
```

---

## Uso

Ejecutar los notebooks en orden:

```bash
jupyter notebook notebooks/01_EDA_limpieza.ipynb
jupyter notebook notebooks/02_modelo_ML.ipynb
jupyter notebook notebooks/03_RAG_pipeline.ipynb
```

Para arrancar la API:

```bash
uvicorn src.api:app --reload
```

---

## Tecnologías

Python · Pandas · scikit-learn · XGBoost · SHAP · LangChain · Qdrant · FastAPI · Jupyter
